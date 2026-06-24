from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_
import torch.nn.functional as F

import copy
import math

from models.ops.modules import MSDeformAttn

class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN)
    Args:
        input_dim: The dimension of input feature.
        hidden_dim: The dimension of intermediate feature.
        output_dim: The dimension of output.
        num_layers: number of layers.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h,
                                            h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    if pos.size(-1) == 2:
        posemb = torch.cat((pos_y, pos_x), dim=-1)
    elif pos.size(-1) == 4:
        w_embed = pos[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        posemb = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos.size(-1)))
    return posemb

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output
    

class BEVDecoderLayer(nn.Module):
    def __init__(self, cfg, d_model=256, d_ffn=1024, dropout=0.1,
                 activation="relu", n_levels=4, n_heads=8, n_points=16):
        super().__init__()
        self.dim = d_model
        
        self.dataset = cfg.DATASET.TRAIN_DATASET
        
        # 3D self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # cross attention
        num_views = cfg.DATASET.CAMERA_NUM
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Frame self-attention
        self.self_attn_frame = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Query self-attention
        self.self_attn_query = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout4 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(d_model)
        
        # 3D cross-attend views
        self.cross_attn_views = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout5 = nn.Dropout(dropout)
        self.norm5 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.view_gate_pooling = nn.Sequential(
            nn.Linear(d_model, 1)
        )

        self.original_size = cfg.DATASET.ORIGINAL_SIZE
        self.image_size = cfg.DATASET.IMAGE_SIZE
        self.worldgrid_size = cfg.DATASET.WORLDGRID
        self.translation_scale = cfg.DATASET.TRANSLATION_SCALE
        

        num_classes = 2
        self.bev_class_embed = nn.Linear(d_model, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.bev_class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        self.img_class_embed = nn.Linear(d_model, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.img_class_embed.bias.data = torch.ones(1) * bias_value
        
        self.cyclinder_embed = MLP(d_model, d_model, 4, 3)
        self.cyclinder_sigma = nn.Linear(d_model, 2, bias=False)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def absolute2norm(self, absolute_coords, worldgrid_size, space="bev"):
        device = absolute_coords.device
        if space == "bev":
            grid_size = torch.tensor(worldgrid_size).to(device)
            norm_coords = absolute_coords / grid_size
        elif space == "bev2d":
            grid_size = torch.tensor(worldgrid_size[:2]).to(device)
            norm_coords = absolute_coords / grid_size
        elif space == "img":
            grid_size = torch.tensor(self.original_size).to(device)
            norm_coords = absolute_coords / grid_size
        return norm_coords

    def norm2absolute(self, norm_coords, worldgrid_size, space="bev"):
        device = norm_coords.device
        if space == "bev":
            grid_size = torch.tensor(worldgrid_size).to(device)
            absolute_coords = norm_coords * grid_size
        elif space == "bev2d":
            grid_size = torch.tensor(worldgrid_size[:2]).to(device)
            absolute_coords = norm_coords * grid_size
        elif space == "img":
            grid_size = torch.tensor(self.original_size).to(device)
            absolute_coords = norm_coords * grid_size
        return absolute_coords

    
    def project_img_points(self, img_points, meta, cam_order, batch_size, nbins, device):
        img_points_norm = img_points.contiguous().view(batch_size*nbins, 2)
        img_points_absolute = self.norm2absolute(img_points_norm, "img")
        
        img_points_absolute_homo = torch.cat([img_points_absolute, torch.ones((img_points_absolute.shape[0], 1), 
                                                    dtype=img_points_absolute.dtype, device=device)], dim=1)
        img_points_projbev_xy_absolute = torch.matmul(meta['bev_T_pix'][0][cam_order], img_points_absolute_homo.transpose(0,1)).transpose(0,1)
        img_points_projbev_xy_absolute = img_points_projbev_xy_absolute / img_points_projbev_xy_absolute[:,2:3]
        img_points_projbev_xy_absolute = img_points_projbev_xy_absolute[:,:2]
        
        img_points_projbev_xy_absolute = img_points_projbev_xy_absolute.view(batch_size, nbins, 2)
        
        return img_points_projbev_xy_absolute

    def get_camera_center(self, E):
        """
        Compute the camera center C = -R^T * t
        R: Rotation matrix (3x3)
        t: Translation vector (3x1 or (3,))
        """
        R = E[:3,:3]
        t = E[:3,3:] * self.translation_scale
        C = -R.T @ t
        return C.flatten()

    def my_cylinder_tangent_edges(self, cylinder_queries, camera_origin, flip=False):
        """
        Compute two vertical 3D edges tangent to the cylinder from the camera's perspective.
        The cylinder is assumed to be upright along the Z-axis and z=0 at base.

        center_xy: (x, y) of cylinder center in world coords
        radius: radius of the cylinder
        height: height of the cylinder
        camera_origin: (3,) array of camera position in world coords
        """
        
        center_xyz = torch.cat([cylinder_queries[...,0:2], torch.zeros_like(cylinder_queries[...,0:1])], dim=-1)
        radius = cylinder_queries[..., 2:3]
        height = cylinder_queries[..., 3:4]
        
        # Project to XY plane
        v = center_xyz - camera_origin
        
        # Orthogonal vector in XY
        v_xy = v[...,:2]
        orthogonal = torch.stack([-v_xy[..., 1], v_xy[..., 0]], dim=-1)  # [-y, x] → Shape: [B, 2]
        norm = torch.linalg.norm(orthogonal, dim=-1, keepdim=True).clamp_min(1e-8)  # Avoid divide by zero
        
        orthonormal_xy = orthogonal / norm  # Shape: [B, 2]
        orthonormal = torch.cat([orthonormal_xy, torch.zeros_like(orthonormal_xy[...,0:1])], dim=-1)  # [B, 3]
    
        # Point on orthonormal line from p1 at distance 'distance'
        if self.dataset == 'wildtrack' or self.dataset == 'gmvd':
            left_bottom_point_ = center_xyz + radius * orthonormal
            right_bottom_point_ = center_xyz - radius * orthonormal
        elif  self.dataset == 'multiviewx':
            left_bottom_point_ = center_xyz - radius * orthonormal
            right_bottom_point_ = center_xyz + radius * orthonormal
        
        if flip:  # swap points
            left_bottom_point = right_bottom_point_
            right_bottom_point = left_bottom_point_
        else:
            left_bottom_point = left_bottom_point_ 
            right_bottom_point = right_bottom_point_
        
        left_bottom_point = torch.cat([left_bottom_point[...,:2],torch.zeros_like(left_bottom_point[...,0:1])], dim=-1)
        right_bottom_point = torch.cat([right_bottom_point[...,:2],torch.zeros_like(right_bottom_point[...,0:1])], dim=-1)
        left_top_point = torch.cat([left_bottom_point[...,:2],torch.ones_like(left_bottom_point[...,0:1])*height], dim=-1)
        
        return torch.cat([left_top_point[:,None,:,:], right_bottom_point[:,None,:,:]],dim=1)
    

    def forward(self, layer, tgt, memories, query_bbox, grid_size, src_views,
                src_spatial_shapes, level_start_index, src_valid_ratios, src_padding_mask=None, 
                meta = None, threshold=0.5):
        '''
        During each decoder layer, each 3D poses of each queries will be projected into each camera view to aggregate features and
        update coarse projected 2D poses. Then, a triangulation will be performed to get updated 3D poses for each queries. Also,
        the feature of each query will be updated with the aggregated features. 

        Args:
            @tgt: query features
            @query_pos: query position embeddings
            @reference_points: 3D points of queries
            @src_views: source views
            @src_spatial_shapes: spatial shapes of source views
            @level_start_index: start index of each level
            @meta: meta information of cameras
        '''
        
        batch_size = tgt.shape[0]
        num_queries = tgt.shape[1]
        device = tgt.device
        # h, w = src_spatial_shapes[0]
        nfeat_level = len(src_spatial_shapes)    # 3
        nviews = len(src_views)//batch_size   ## batch, views, ...
        eps=1e-5
        
        # 1. Self-attention on 3D queries
        if layer == 0:
            query_pos = pos2posemb2d(query_bbox[..., :2],128)  # only x,y ; no z
            q = k = self.with_pos_embed(tgt, query_pos)
        else:
            q = k = tgt
        
        query_pos = pos2posemb2d(query_bbox[..., :2],128)  # only x,y ; no z
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)           # torch.Size([3, 768, 256])
        
        # 2. Generate anchor points
        original_xy_absolute = torch.cat([query_bbox[..., :2], torch.zeros_like(query_bbox[..., :2][...,0:1])], dim=-1)
        original_rh_absolute = query_bbox[..., 2:]
        sampling_3D_points_absolute = original_xy_absolute + torch.cat([torch.zeros_like(original_rh_absolute),original_rh_absolute[...,1:2]/2], dim=-1)  # torch.Size([1, 1024, 3])
        
        # 3. projective attention to extract features from mulitple views
        tgt_views = tgt.expand(nviews, -1,-1,-1).permute(1,0,2,3).reshape(batch_size*nviews, num_queries, self.dim) # nviews adaptive        
        sampling_3D_points_homo = torch.cat([sampling_3D_points_absolute, torch.ones_like(sampling_3D_points_absolute[..., :1])], dim=-1)
        proj_matrices = meta['pix_T_bev'].unsqueeze(2)  # torch.Size([3, 7, 1, 4, 4])
        sampling_3D_points_homo = sampling_3D_points_homo.unsqueeze(1).unsqueeze(-1) # torch.Size([3, 1, 768, 4, 1])
        sampling_2D_points_homo = torch.matmul(proj_matrices, sampling_3D_points_homo).squeeze(-1)  # torch.Size([3, 7, 768, 4])
        sampling_2D_points_homo = sampling_2D_points_homo / sampling_2D_points_homo[...,2:3]
        sampling_2D_points_homo_nonzero = torch.maximum(sampling_2D_points_homo, torch.zeros_like(sampling_2D_points_homo) + eps)
        sampling_2D_points_absolute = sampling_2D_points_homo_nonzero[...,:2]       # torch.Size([3, 7, 768, 2])
        sampling_2D_points_norm = self.absolute2norm(sampling_2D_points_absolute, meta['grid_shape'], "img")
        sampling_2D_points_norm = torch.clamp(sampling_2D_points_norm, -0.01, 1.01).view(batch_size*nviews, num_queries, 2)     # torch.Size([3* 7, 768, 2])
        sampling_2D_points_norm_expand = (sampling_2D_points_norm[:, :, None] * src_valid_ratios[:, None])  # torch.Size([3*7, 768, 3, 2])
        
        validity_mask =   (sampling_2D_points_norm[..., 0] >= 0.0) \
                        & (sampling_2D_points_norm[..., 1] >= 0.0) \
                        & (sampling_2D_points_norm[..., 0] <= 1.0) \
                        & (sampling_2D_points_norm[..., 1] <= 1.0)  # like a mask   torch.Size([3*7, 768])
        validity_mask = validity_mask.view(batch_size, nviews, num_queries)
        
        tgt_views2 = self.cross_attn(tgt_views,
                            sampling_2D_points_norm_expand,
                            memories, src_spatial_shapes, level_start_index, None)
        tgt_views = tgt_views + self.dropout2(tgt_views2)
        tgt_views = self.norm2(tgt_views)   # torch.Size([3*7, 768, 256])
        
        # 5. Factorized Self-attention: Different Queries on 1 View -> Same Queries on Multiple Views
            # Flatten validity to match flattening in attention mask (Q-major)
        num_heads = self.self_attn_query.num_heads
        mask_q = validity_mask.unsqueeze(-1)           # (B, F, Q, 1)
        mask_k = validity_mask.unsqueeze(-2)           # (B, F, 1, Q)
        valid_mask = mask_q | mask_k           # (B, F, Q, Q)
        valid_mask = ~valid_mask
        valid_mask = valid_mask.view(batch_size * nviews, num_queries, num_queries)
        valid_mask = valid_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        valid_mask = valid_mask.view(batch_size * nviews * num_heads, num_queries, num_queries)
        
            # Intra-view attention
        tgt_views_frame, _ = self.self_attn_frame(tgt_views, tgt_views, 
                                                  value=tgt_views, attn_mask = valid_mask)
        tgt_views = tgt_views + self.dropout3(tgt_views_frame)
        tgt_views = self.norm3(tgt_views)       # torch.Size([21, 768, 256]) 
            # Inter-view attention
        tgt_views = tgt_views.view(batch_size, nviews, num_queries, self.dim).permute(0,2,1,3).flatten(0,1)
        tgt_views_query, _ = self.self_attn_query(tgt_views, tgt_views, 
                                                  value=tgt_views)
        tgt_views = tgt_views + self.dropout4(tgt_views_query)
        tgt_views = self.norm4(tgt_views)
        
        tgt_views = tgt_views.view(batch_size, num_queries, nviews, self.dim)   # torch.Size([3, 768, 7, 256])
        img_class = self.img_class_embed(tgt_views).permute(0, 2, 1, 3)     # torch.Size([3, 7, 768, 2])
        
        # 6. Aggregate 3D features from 2D view features
        tgt_3D_weights = self.view_gate_pooling(tgt_views)
        tgt_3D_weights = torch.softmax(tgt_3D_weights, dim=2)  # softmax over frames
        tgt_3D_update = torch.sum(tgt_3D_weights * tgt_views, dim=2)  # [B, Q, C]
        
        tgt_update = tgt + self.dropout5(tgt_3D_update)
        tgt_update = self.norm5(tgt_update) # torch.Size([1, 1024, 256])
        tgt_update = self.forward_ffn(tgt_update)
        
        bev_class = self.bev_class_embed(tgt_update)    # torch.Size([3, 768, 2])
        
        # 7. Update 3D cylinder
        bev_sigma = self.cyclinder_sigma(tgt_update).sigmoid()
        tmp = self.cyclinder_embed(tgt_update)
        query_bbox_update = query_bbox + tmp
        
        # 8. Project 3D cylinder to 2D rectangular bbox / Indirect constraint            
            # project 3D cylinder to 2D polygon boxes
        projected_2D_bboxes = []
        for bdx in range(batch_size):
            projected_2D_views = []
            for n in range(nviews):
                camera_origin = self.get_camera_center(meta['cam_T_global'][bdx][n])
                
                cylinder_corners_absolute = self.my_cylinder_tangent_edges(query_bbox_update[bdx:bdx+1], camera_origin)
                cylinder_corners_absolute = cylinder_corners_absolute.reshape(1*2*num_queries, 3)  # left, right
                # convert points to this view
                cylinder_corners_homo = torch.cat([cylinder_corners_absolute, torch.ones_like(cylinder_corners_absolute[..., :1])], dim=-1)
                cylinder_proj_corners_homo = torch.matmul(meta['pix_T_bev'][bdx][n], cylinder_corners_homo.transpose(0,1)).transpose(0,1)
                cylinder_proj_corners_homo[cylinder_proj_corners_homo == 0] = 1e-8
                cylinder_proj_corners_homo = cylinder_proj_corners_homo / cylinder_proj_corners_homo[:,2:3]
                cylinder_proj_corners_absolute = cylinder_proj_corners_homo[:,:2]
                
                cylinder_proj_corners_absolute = cylinder_proj_corners_absolute.view(1, 2, num_queries, 2).permute(0,2,1,3)    # left, right
                cylinder_proj_corners_norm = self.absolute2norm(cylinder_proj_corners_absolute, meta['grid_shape'], "img").reshape(1, num_queries, 4)
        
                cylinder_proj_corners_norm = torch.clamp(cylinder_proj_corners_norm, min=torch.zeros(4).to(device), max=torch.ones(4).to(device)) # avoid zero outputs
                projected_2D_views.append(cylinder_proj_corners_norm)
        
            projected_2D_views = torch.stack(projected_2D_views)      # xyxy  # torch.Size([7, 1, 768, 4])
            projected_2D_bboxes.append(projected_2D_views)
        projected_2D_bboxes= torch.cat(projected_2D_bboxes,dim=1).permute(1,0,2,3)
        
        return tgt_update, query_bbox_update, projected_2D_bboxes, bev_class, img_class, bev_sigma


class BEVDecoder(nn.Module):
    def __init__(self, cfg, decoder_layer,
                 num_layers, return_intermediate=True):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memories, src_views,
                src_spatial_shapes, src_level_start_index, src_valid_ratios, meta, 
                query_bbox, grid_size, src_padding_mask=None, threshold = 0.5):
        
        bev_classes = []
        img_classes = []
        intermediate_bev_coords = []
        intermediate_box_coords = []
        intermediate_rh_shapes = []
        bev_sigmas = []
        output = tgt
        for lid, layer in enumerate(self.layers):
            
            output, query_bbox_update, proj_bboxes, bev_class, img_class, bev_sigma = layer(
                            lid, output, memories, query_bbox , grid_size,
                            src_views, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                            src_padding_mask, meta, threshold=threshold)
            query_bbox = query_bbox_update.detach()
            
            bev_classes.append(bev_class)
            img_classes.append(img_class)
            intermediate_bev_coords.append(query_bbox_update[...,:2])
            intermediate_rh_shapes.append(query_bbox_update[...,2:])
            intermediate_box_coords.append(proj_bboxes)
            bev_sigmas.append(bev_sigma)

        return torch.stack(bev_classes), torch.stack(img_classes), \
                torch.stack(intermediate_bev_coords), torch.stack(intermediate_rh_shapes), \
                torch.stack(intermediate_box_coords), torch.stack(bev_sigmas)

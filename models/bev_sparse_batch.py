from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_, normal_
import torch.distributed as dist
import torch.nn.functional as F

import utils.geom
import utils.vox
import utils.basic

import copy
import math
from models.encoder import Encoder_fasterrcnn_resnet50_fpn
from models.VIT_encoder import CoDETR
from models.bev_sparse_batch_decoder import BEVDecoder, BEVDecoderLayer
from models.bev_sparse_batch_decoder import DeformableTransformerEncoder, DeformableTransformerEncoderLayer

from models.position_encoding import PositionEmbeddingSine
from models.matcher import HungarianMatcher

from torchvision.ops.boxes import box_area

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter + 1e-6

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] +1e-6

    return iou - (area - union) / area

def get_model(cfg, is_train=True, fix_backbone=True, input_size=(640,360)):
    backbone = Encoder_fasterrcnn_resnet50_fpn(256)
    # backbone = CoDETR()
    

    if fix_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        print(' * Fix backbone: True')
    else:
        for param in backbone.parameters():
            param.requires_grad = True
        print(' * Fix backbone: False')

    model = BEVGEO(backbone, cfg)
    return model

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


class BEVGEO(nn.Module):
    def __init__(self, 
                 backbone,
                 cfg,
                 device=torch.device('cuda')):
        super().__init__()
        self.cfg = cfg
        self.max_objects = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.backbone = backbone.to(device)
        
        self.latent_dim = cfg.DECODER.d_model
        self.num_cameras = cfg.DATASET.CAMERA_NUM
        self.num_queries = cfg.DECODER.num_instance
        self.original_size = cfg.DATASET.ORIGINAL_SIZE
        self.image_size = cfg.DATASET.IMAGE_SIZE
        self.worldgrid_size = cfg.DATASET.WORLDGRID # [480, 1440, 80]
        self.use_feat_level = cfg.DECODER.use_feat_level
        
        N_steps = cfg.DECODER.d_model // 2
        self.pos_encoding = PositionEmbeddingSine(N_steps, normalize=True)
        
        encoder_layer = DeformableTransformerEncoderLayer(cfg.DECODER.d_model, cfg.DECODER.dim_feedforward,
                                                          cfg.DECODER.dropout, cfg.DECODER.activation,
                                                          4, #cfg.DECODER.num_feature_levels,   # 3
                                                          cfg.DECODER.nhead, cfg.DECODER.dec_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, 4)
        
        decoder_layer = BEVDecoderLayer(cfg, 
                        d_model=cfg.DECODER.d_model, 
                        d_ffn=cfg.DECODER.dim_feedforward,
                        dropout=cfg.DECODER.dropout,
                        activation=cfg.DECODER.activation, 
                        n_levels=4,    # 3
                        n_heads=cfg.DECODER.nhead, 
                        n_points=cfg.DECODER.dec_n_points)
        self.decoder = BEVDecoder(cfg, decoder_layer, 
                                cfg.DECODER.num_decoder_layers,
                                cfg.DECODER.return_intermediate_dec)
        
        matcher = HungarianMatcher(cfg)
        
        
        self.weight_dict = {
            'loss_ce': 4, 'loss_bev': 20, 'loss_bbox': 4, 'loss_giou': 2, 
            'loss_ce_0': 1, 'loss_bev_0': 5, 'loss_bbox_0': 1, 'loss_giou_0': 1, 
            'loss_ce_1': 1, 'loss_bev_1': 5, 'loss_bbox_1': 1, 'loss_giou_1': 1, 
            'loss_ce_2': 1, 'loss_bev_2': 5, 'loss_bbox_2': 1, 'loss_giou_2': 1, 
        }
        
        losses = ['labels', 'bev', 'img']   # bev_loss does not contribute to the total loss
        
        focal_alpha = 0.25
        num_classes = 2
        self.criterion = SetCriterion(num_classes, matcher, self.weight_dict,
                                      losses, cfg,
                                      focal_alpha=focal_alpha)
        self.criterion.to(device)
        
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
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

    
    def create_pos_embedding(self, img_size, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        H, W = img_size
        not_mask = torch.ones([1, H, W])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def initialize_points(self, device, meta):
        def get_divisors(n):
            """Return all positive divisors of n."""
            divisors = []
            i = 1
            while i * i <= n:
                if n % i == 0:
                    divisors.append(i)
                    if i * i != n:
                        divisors.append(n // i)
                i += 1
            return divisors


        def find_best_xy_fast(expected_ratio, product=768):
            """
            Fast version: only tries valid divisors of `product`.
            """
            divisors = get_divisors(product)
            best_x, best_y = None, None
            best_error = float("inf")

            for x in divisors:
                y = product // x
                ratio = x / y
                error = abs(ratio - expected_ratio)

                if error < best_error:
                    best_error = error
                    best_x, best_y = x, y

            return best_x, best_y, best_x / best_y
        
        grid_shape = meta['grid_shape']
        expected_ratio = grid_shape[0].tolist()[0] / grid_shape[1].tolist()[0]
        num_x, num_y, true_ratio = find_best_xy_fast(expected_ratio, self.num_queries)
        
        x_ = torch.linspace(0., 1., num_x)
        y_ = torch.linspace(0., 1., num_y)
        
        x, y = torch.meshgrid(x_, y_)      # torch.Size([32, 32]), torch.Size([32, 32]) # combine them to make grid        
        root_coordinates = torch.cat([x.unsqueeze(-1),y.unsqueeze(-1)], dim=-1)
        root_coordinates = root_coordinates.view(-1, 2)     # torch.Size([1024, 2])
        root_coordinates_absolute = self.norm2absolute(root_coordinates, grid_shape, "bev2d" )
        
        r = (torch.ones([self.num_queries])*15.0).unsqueeze(-1)
        h = (torch.ones([self.num_queries])*65.0).unsqueeze(-1)
        
        ref_cylinders = torch.cat([root_coordinates_absolute, r, h] ,dim=1).float().to(device)  # torch.Size([768,4])
        
        return ref_cylinders, num_x, num_y
        
    def forward(self, views=None, meta=None, targets=None, is_train=False):
        # all_feats = self.backbone(views[0],self.use_feat_level)
        batch, nviews, img_c, img_h, img_w = views.shape
        device = views.device
        
        all_feats = self.backbone(views.view(batch*nviews,img_c,img_h,img_w))
        all_feats = [all_feats[i] for i in [0,1,2,3]]
        
        src_flatten_views = [] 
        mask_flatten_views = [] 
        spatial_shapes_views = [] # store feature map size for each feature level
        raw_mask_views = []
        pos_views = []
        for lvl, src in enumerate(all_feats):
            b_v, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_views.append(spatial_shape)
            
            mask = src.new_zeros(b_v, h, w).bool()  # an empty mask
            raw_mask_views.append(mask)
            mask = mask.flatten(1) # get vector feature for each batch
            mask_flatten_views.append(mask)
            
            B, C, H, W = src.shape
            pos_embed = self.create_pos_embedding([H, W], C//2).to(src.device).repeat(B,1,1,1) # unsqueeze(0).repeat(B,1,1,1)  # [C, H, W]
            pos_embed = pos_embed.flatten(2).transpose(1,2)
            pos_views.append(pos_embed)
            
            src = src.flatten(2).transpose(1, 2)
            src_flatten_views.append(src)
        
        src_flatten_views = torch.cat(src_flatten_views, 1)
        spatial_shapes_views = \
            torch.as_tensor(spatial_shapes_views,
                            dtype=torch.long,
                            device=mask.device)
        pos_views = torch.cat(pos_views, dim=1)
                
        level_start_index_views = torch.cat((spatial_shapes_views.new_zeros(1),
                               spatial_shapes_views.prod(1).cumsum(0)[:-1]))

        valid_ratios_views = torch.stack([self.get_valid_ratio(m)
                                          for m in raw_mask_views], 1)  # torch.Size([7, 3, 2])
        
        mask_flatten_views = [m.flatten(1) for m in mask_flatten_views]
        mask_flatten_views = torch.cat(mask_flatten_views, dim=1)
        
        # Encode images         
        memories = self.encoder(src_flatten_views, spatial_shapes_views, level_start_index_views, 
                            valid_ratios_views, pos_views, mask_flatten_views)      # torch.Size([7, 75600, 256])
        
        query_3D_bbox, num_x, num_y = self.initialize_points(device, meta)
        query_3D_bbox = query_3D_bbox.expand(batch, -1, -1)  # add batch size = 1
        tgt = torch.zeros([batch, self.num_queries,self.latent_dim],device=query_3D_bbox.device)
        
        # use DAB, query_pos = None
        bev_classes, img_classes, inter_references_bev_pts, inter_references_rh, inter_references_img_boxes, bev_vars = \
            self.decoder(tgt, memories, 
                         src_views=src_flatten_views,
                         src_spatial_shapes=spatial_shapes_views,
                         src_level_start_index=level_start_index_views,
                         src_valid_ratios=valid_ratios_views,
                         meta=meta,
                         query_bbox=query_3D_bbox,
                         grid_size = (num_x, num_y),
                         src_padding_mask=mask_flatten_views,
                         threshold = 0.3)
        
        inter_references_bev_pts = torch.nan_to_num(inter_references_bev_pts)
        inter_references_rh = torch.nan_to_num(inter_references_rh)
        inter_references_img_boxes = torch.nan_to_num(inter_references_img_boxes)
        bev_classes = torch.nan_to_num(bev_classes)
        img_classes = torch.nan_to_num(img_classes)
        bev_vars = torch.nan_to_num(bev_vars)
        
        # Final layer output
        out_dict = {'pred_logits': bev_classes[-1],
                'pred_img_logits': img_classes[-1],
               'pred_bev': inter_references_bev_pts[-1],
               'pred_rh': inter_references_rh[-1],
               'pred_boxes': inter_references_img_boxes[-1],
               'bev_vars': bev_vars[-1]}

        if is_train:
            out_dict['aux_outputs'] = [{'pred_logits': a, 'pred_img_logits': b, 'pred_bev':c, 'pred_boxes': d, 'bev_vars':e} \
                for a, b, c, d, e in zip(bev_classes[:-1], img_classes[:-1], inter_references_bev_pts[:-1], inter_references_img_boxes[:-1], bev_vars[:-1])]

            loss_dict = self.criterion(out_dict, meta, targets)
            total_loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)

            return out_dict, total_loss, loss_dict
        else:
            return out_dict

class SetCriterion(torch.nn.Module):
    """
    The process happens in two steps:
        1) we compute hungarian assignment
        between ground truth poses and the outputs of the model
        2) we supervise each pair of matched
        ground-truth / prediction (supervise class and pose)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, cfg, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories,
            omitting the special no-object category

            matcher: module able to compute a
            matching between targets and proposals

            weight_dict: dict containing as key the names of
            the losses and as values their relative weight.

            losses: list of all the losses to be applied.
            See get_loss for list of available losses.

            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        # self.img_size = cfg.NETWORK.IMAGE_SIZE
        
        self.original_size = cfg.DATASET.ORIGINAL_SIZE
        self.image_size = cfg.DATASET.IMAGE_SIZE
        self.worldgrid_size = cfg.DATASET.WORLDGRID
       
        self.eos_coef = 0.1
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

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

    def loss_labels(self, outputs, meta, targets, indices, num_samples, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key
        "labels" containing a tensor of dim [nb_target_poses]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # torch.Size([3, 768, 2])
        
        idx = self._get_src_permutation_idx(indices)
        device = src_logits.device
        target_classes_o = torch.ones((num_samples,)).long().to(device)   # torch.Size([26])
        target_classes = torch.full(src_logits.shape[:2],
                                    self.num_classes,
                                    dtype=torch.int64,
                                    device=src_logits.device)   #  torch.Size([1, 768])
        target_classes[idx] = target_classes_o
        
        target_classes_onehot = \
            torch.zeros([src_logits.shape[0],
                         src_logits.shape[1],
                         src_logits.shape[2] + 1],
                        dtype=src_logits.dtype,
                        layout=src_logits.layout,
                        device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits,
                                     target_classes_onehot,
                                     num_samples,
                                     alpha=self.focal_alpha,
                                     gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        
        return losses

    def loss_bboxes(self, outputs, meta, targets, indices, num_samples):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        
        """
        Rotated IoU box loss
        """
        
        bs, num_views, num_queries, _ = outputs['pred_boxes'].shape
        
        idx = self._get_src_permutation_idx(indices)
        out_bbox = outputs['pred_boxes'].permute(0,2,1,3)[idx]  # torch.Size([102, 7, 4])
        tgt_box = torch.cat([t[i] for t, (_, i) in zip(targets['img_pts'].permute(0,2,1,3), indices)], dim=0)   # torch.Size([102, 7, 4])
        tgt_mask = torch.cat([t[i] for t, (_, i) in zip(targets['img_pids'].permute(0,2,1), indices)], dim=0)   # torch.Size([102, 7])
        tgt_mask = tgt_mask!=-1
        
        out_bbox_flatten = out_bbox.flatten(0,1)
        base_x_mask = out_bbox_flatten[:, 2] >= out_bbox_flatten[:, 0]    # x correct
        base_y_mask = out_bbox_flatten[:, 3] >= out_bbox_flatten[:, 1]    # y correct
        error_mask = (base_y_mask & ~base_x_mask) | (base_y_mask & base_x_mask)
        flip_mask = (base_y_mask & ~base_x_mask) | (~base_y_mask & ~base_x_mask)
        flip_pos = torch.where(flip_mask==True)[0]
        if len(flip_pos) > 0:
            for fdx in range(len(flip_pos)):
                out_bbox_flatten[flip_pos[fdx]] = out_bbox_flatten[flip_pos[fdx]][[2,1,0,3]]
        
        
        
        error_mask = (error_mask*tgt_mask.flatten().float()).unsqueeze(-1)
        sizes = [len(indices[idx][0]) for idx in range(len(indices))]

        losses = {} 
        loss_bbox = F.l1_loss(box_xyxy_to_cxcywh(out_bbox_flatten), tgt_box.flatten(0,1), reduction='none').mul(error_mask)
        
        
        loss_bbox_final = 0
        curr_size = 0
        for idx in range(len(sizes)):
            num_samples = torch.count_nonzero(tgt_mask[curr_size:curr_size+sizes[idx]])
            loss_bbox_final += loss_bbox[curr_size*num_views:(curr_size+sizes[idx])*num_views].sum()/num_samples
            curr_size += sizes[idx]
        
        losses['loss_bbox'] = loss_bbox_final
        
        loss_giou = (1 - torch.diag( generalized_box_iou(
            out_bbox_flatten.mul(error_mask),
            box_cxcywh_to_xyxy(tgt_box.flatten(0,1)).mul(error_mask) ) ) ).mul(tgt_mask.flatten())
        
        loss_giou_final = 0
        curr_size = 0
        # import ipdb; ipdb.set_trace()
        for idx in range(len(sizes)):
            num_samples = torch.count_nonzero(tgt_mask[curr_size:curr_size+sizes[idx]])
            loss_giou_final += loss_giou[curr_size*num_views:(curr_size+sizes[idx])*num_views].sum()/num_samples
            curr_size += sizes[idx]
        
        losses['loss_giou'] = loss_giou_final
        
        return losses
    
    # @torch.no_grad()
    def loss_points(self, outputs, meta, targets, indices, num_samples):
        """Compute the losses related to the bounding poses,
        the L1 regression loss and the GIoU loss
           targets dicts must contain the key "poses"
           containing a tensor of dim [nb_target_poses, 4]
           The target poses are expected in format
           (center_x, center_y, h, w), normalized by the image size.
        """
        # assert 'pred_poses' in outputs      
        idx = self._get_src_permutation_idx(indices) # batch, src
        out_bev = outputs["pred_bev"][idx][...,:2]                 # torch.Size([1, 1024, 2])
        output_var  = outputs["bev_vars"][idx]

        tgt_bev_norm = torch.cat([t[i] for t, (_, i) in zip(targets['bev_pts'], indices)], dim=0)
        
        out_bev_norm = self.absolute2norm(out_bev,meta['grid_shape'],"bev2d")
        
        l1_loss = F.l1_loss(out_bev_norm, tgt_bev_norm, reduction='none') 
        loss_bev = (torch.exp(-output_var) * l1_loss + 0.5 * output_var)

        sizes = [len(indices[idx][0]) for idx in range(len(indices))]
        loss_bev_final = 0
        curr_size = 0
        for idx in range(len(sizes)):
            loss_bev_final += loss_bev[curr_size:curr_size+sizes[idx]].sum()/sizes[idx]
            curr_size += sizes[idx]
        losses = {}
        
        losses['loss_bev'] = loss_bev_final
        
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, meta, targets, indices, num_samples, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'bev': self.loss_points,
            'img': self.loss_bboxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, meta, targets, indices, num_samples, **kwargs)

    def forward(self, outputs, meta, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors,
             see the output specification of the model for the format

             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the
                      losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the
        # outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, meta, targets)

        # Compute the average number of target
        # poses accross all nodes, for normalization purposes
        num_samples = sum(len(t[0]) for t in indices)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, meta, targets, indices, num_samples, **kwargs))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, meta, targets)
                aux_num_samples = sum(len(t[0]) for t in indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, meta, targets, indices, aux_num_samples, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
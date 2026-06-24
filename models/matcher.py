import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class HungarianMatcher(nn.Module):
    """
    Computes an assignment between model predictions and ground-truth targets
    using the Hungarian algorithm with a combined BEV + classification + box cost.
    """

    def __init__(self, cfg,
                 cost_class: float = 1,
                 cost_bev: float = 2,
                 cost_giou: float = 1,
                 cost_bbox: float = 2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bev = cost_bev
        self.cost_giou = cost_giou
        self.cost_bbox = cost_bbox

        self.original_size = cfg.DATASET.ORIGINAL_SIZE
        self.image_size = cfg.DATASET.IMAGE_SIZE
        self.worldgrid_size = cfg.DATASET.WORLDGRID

        assert cost_class != 0 or cost_bev != 0 or cost_giou != 0, \
            "all costs cant be 0"

    def absolute2norm(self, absolute_coords, worldgrid_size, space="bev"):
        device = absolute_coords.device
        if space == "bev":
            grid_size = torch.tensor(worldgrid_size).to(device)
        elif space == "bev2d":
            grid_size = torch.tensor(worldgrid_size[:2]).to(device)
        elif space == "img":
            grid_size = torch.tensor(self.original_size).to(device)
        return absolute_coords / grid_size

    def norm2absolute(self, norm_coords, worldgrid_size, space="bev"):
        device = norm_coords.device
        if space == "bev":
            grid_size = torch.tensor(worldgrid_size).to(device)
        elif space == "bev2d":
            grid_size = torch.tensor(worldgrid_size[:2]).to(device)
        elif space == "img":
            grid_size = torch.tensor(self.original_size).to(device)
        return norm_coords * grid_size

    def forward(self, outputs, meta, targets):
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            device = outputs["pred_logits"].device

            # BEV cost
            out_bev = outputs["pred_bev"].flatten(0, 1)[..., :2]
            tgt_bev = []
            sizes = []
            for bdx in range(bs):
                num_targets = len(torch.where(targets["bev_pids"][bdx] != -1)[0])
                tgt_bev.append(targets["bev_pts"][bdx][:num_targets])
                sizes.append(num_targets)
            tgt_bev = torch.cat(tgt_bev)
            tgt_bev = self.norm2absolute(tgt_bev, meta['grid_shape'], "bev2d").float()
            cost_bev = torch.cdist(out_bev, tgt_bev, p=1) / 100.

            # Classification cost (focal-loss style)
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            tgt_ids = torch.ones(len(tgt_bev)).long()
            alpha, gamma = 0.25, 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Image box cost
            out_box = outputs["pred_boxes"].permute(1, 0, 2, 3).flatten(1, 2)
            tgt_box = []
            tgt_box_id = []
            for bdx in range(bs):
                num_targets = len(torch.where(targets["bev_pids"][bdx] != -1)[0])
                tgt_box.append(targets["img_pts"][bdx].permute(1, 0, 2)[:num_targets])
                tgt_box_id.append(targets['img_pids'][bdx].transpose(0, 1)[:num_targets])
            tgt_box = torch.cat(tgt_box).permute(1, 0, 2)
            tgt_box_id = torch.cat(tgt_box_id).transpose(0, 1)
            tgt_mask = tgt_box_id != -1

            cost_bbox = torch.cdist(
                box_xyxy_to_cxcywh(out_box)[:, :, :2], tgt_box[:, :, :2], p=1
            )
            cost_bbox = cost_bbox * tgt_mask.unsqueeze(1)
            cost_bbox = cost_bbox.sum(dim=0) / tgt_mask.sum(dim=0).clamp(min=1).unsqueeze(0)

            C = self.cost_bev * cost_bev + self.cost_class * cost_class + self.cost_bbox * cost_bbox
            C = C.view(bs, num_queries, -1).cpu()
            C_bev = cost_bev.view(bs, num_queries, -1).cpu()

            C = torch.nan_to_num(C, nan=100.0, posinf=100.0, neginf=-100.0)
            C_bev = torch.nan_to_num(C_bev, nan=100.0, posinf=100.0, neginf=-100.0)

            indices = self.adaptive_hungarian_matching(C, C_bev, sizes, num_iterations=3)

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    def adaptive_hungarian_matching(self, C, C_bev, sizes, num_iterations=3):
        """
        Iterative Hungarian matching: each round removes already-matched rows,
        allowing secondary matches within the distance threshold.
        """
        large_value = 10000
        dist_thres = 25 / 100.
        C_split = C.split(sizes, -1)
        C_bev_split = C_bev.split(sizes, -1)

        output = []
        for batch_id, (C_b, C_bev_b) in enumerate(zip(C_split, C_bev_split)):
            C_batch = C_b[batch_id].clone()
            C_bev_batch = C_bev_b[batch_id].clone()
            matches = [[], []]

            for num_iter in range(num_iterations):
                row_ind, col_ind = linear_sum_assignment(C_batch)
                if num_iter != 0:
                    keep = torch.where(C_bev_batch[[row_ind, col_ind]] <= dist_thres)[0].tolist()
                    row_ind = row_ind[keep]
                    col_ind = col_ind[keep]
                matches[0].extend(row_ind)
                matches[1].extend(col_ind)
                C_batch[row_ind, :] = large_value

            output.append((np.array(matches[0]), np.array(matches[1])))

        return output

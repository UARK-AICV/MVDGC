from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import argparse
import os
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm

from configs.config import config
from configs.config import update_config, update_config_dynamic_input

from models.bev_sparse_batch import get_model
from dataset.wildtrack import Wildtrack
from dataset.multiviewx import MultiviewX
from dataset.gmvd import GMVD
from utils.utils import get_rank
from utils.metrics import CLEAR_MOD_HUN_extended, single_keypoint_nms

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Test Geometry network')
    parser.add_argument('--cfg', help='experiment configure file name',
                        required=True, type=str)
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--model_path', default=None, type=str,
                        help='pass model path for evaluation')
    parser.add_argument('--exp_name', '-n', default='exp', type=str)

    args, unknown = parser.parse_known_args()
    update_config(args.cfg)
    update_config_dynamic_input(unknown)
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    final_output_dir = 'output'

    print('=> Loading data ..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])

    if config.DATASET.TRAIN_DATASET == 'wildtrack':
        test_dataset = Wildtrack(config, transform, istrain=False)
    elif config.DATASET.TRAIN_DATASET == 'multiviewx':
        test_dataset = MultiviewX(config, transform, istrain=False)
    elif config.DATASET.TRAIN_DATASET == 'gmvd':
        test_dataset = GMVD(config, transform, istrain=False)

    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        sampler=sampler_val,
        pin_memory=True,
        num_workers=config.WORKERS)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = get_model(config, True, True)
    model.to(device)

    weight_name = "wildtrack_deform_VIT_epoch_28.pth"
    max_objects = 100
    bev_threshold = 0.6
    nms_radius = 15

    state_dict = torch.load(os.path.join(final_output_dir, weight_name))
    model.load_state_dict(state_dict)

    print('=> Testing...')

    model.eval()
    worldgrid_size = config.DATASET.WORLDGRID
    moda_gt_list = []
    moda_pred_list = []

    with torch.no_grad():
        for frame_id, (inputs, targets, meta) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}
            meta = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in meta.items()}

            num_targets = len(torch.where(targets["bev_pids"][0] != -1)[0])
            target_points_norm = targets['bev_pts'][0][0:num_targets]

            outputs = model(views=inputs, meta=meta, targets=targets,
                            is_train=False)

            output_logits = outputs['pred_logits']
            output_bevs = outputs['pred_bev']

            prob = output_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(
                prob.view(output_logits.shape[0], -1), max_objects, dim=1)
            scores = topk_values
            topk_points = topk_indexes // output_logits.shape[2]

            threshold_range = len(torch.where(scores >= bev_threshold)[1])
            bev_points_absolute = torch.gather(
                output_bevs, 1, topk_points.unsqueeze(-1).repeat(1, 1, 2))
            nms_input = torch.cat(
                [bev_points_absolute[:, :threshold_range, :],
                 scores[:, :threshold_range].unsqueeze(-1)],
                dim=-1).squeeze(0).cpu().numpy()
            keep_inds = single_keypoint_nms(nms_input, nms_radius,
                                            max_objects, True)
            bev_points_absolute = bev_points_absolute[0][keep_inds]
            for idx in range(len(bev_points_absolute)):
                moda_pred_list.append([
                    frame_id,
                    bev_points_absolute[idx][0].item(),
                    bev_points_absolute[idx][1].item(),
                ])

            worldgrid_tensor = torch.tensor(worldgrid_size[:2]).to(device)
            target_points_absolute = target_points_norm * worldgrid_tensor
            for jdx in range(num_targets):
                moda_gt_list.append([
                    frame_id,
                    target_points_absolute[jdx][0].item(),
                    target_points_absolute[jdx][1].item(),
                    targets["bev_pids"][0][jdx].item(),
                ])

    gtRaw = np.array(moda_gt_list)
    detRaw = np.array(moda_pred_list)
    frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
    frame_ctr = 0
    gt_flag = det_flag = True
    gtAllMatrix = detAllMatrix = 0

    if detRaw is None or detRaw.shape[0] == 0:
        print("No detections — metrics are 0.")
        return

    for t in frames:
        idx = np.where(gtRaw[:, 0] == t)[0]
        idx_len = len(idx)
        tmp = np.zeros((idx_len, 5))
        tmp[:, 0] = frame_ctr
        tmp[:, 1] = np.arange(idx_len)
        tmp[:, 2] = gtRaw[idx, 1]
        tmp[:, 3] = gtRaw[idx, 2]
        tmp[:, 4] = gtRaw[idx, 3]
        gtAllMatrix = tmp if gt_flag else np.concatenate((gtAllMatrix, tmp))
        gt_flag = False

        idx = np.where(detRaw[:, 0] == t)[0]
        idx_len = len(idx)
        tmp = np.zeros((idx_len, 4))
        tmp[:, 0] = frame_ctr
        tmp[:, 1] = np.arange(idx_len)
        tmp[:, 2] = detRaw[idx, 1]
        tmp[:, 3] = detRaw[idx, 2]
        detAllMatrix = tmp if det_flag else np.concatenate((detAllMatrix, tmp))
        det_flag = False
        frame_ctr += 1

    recall, precision, MODA, MODP, matched_pairs = CLEAR_MOD_HUN_extended(
        gtAllMatrix, detAllMatrix)

    print("\t", weight_name)
    print("     Threshold:", bev_threshold, "Max objects:", max_objects,
          "NMS radius:", nms_radius)
    print("==========================")
    print("Recall:", recall)
    print("Precision:", precision)
    print("* MODA:", MODA)
    print("MODP:", MODP)
    print("==========================")


if __name__ == '__main__':
    main()

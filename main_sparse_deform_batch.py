from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
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
from utils.metrics import CLEAR_MOD_HUN, single_keypoint_nms

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def parse_args():
    parser = argparse.ArgumentParser(description='Train Geometry network')
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


def match_name_keywords(n, name_keywords):
    return any(b in n for b in name_keywords)


def get_optimizer(model_without_ddp, weight_decay, optim_type):
    lr = config.TRAIN.LR
    if model_without_ddp.backbone is not None:
        for params in model_without_ddp.backbone.parameters():
            params.requires_grad = False

    lr_linear_proj_names = ['reference_points', 'sampling_offsets']
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if not match_name_keywords(n, lr_linear_proj_names)
                       and p.requires_grad],
            "lr": lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, lr_linear_proj_names)
                       and p.requires_grad],
            "lr": lr * config.DECODER.lr_linear_proj_mult,
        },
    ]

    if optim_type == 'adam':
        return optim.Adam(param_dicts, lr=lr)
    elif optim_type == 'adamw':
        return optim.AdamW(param_dicts, lr=lr, weight_decay=1e-4)


def main():
    args = parse_args()
    device = torch.device(args.device)

    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    final_output_dir = 'output'
    exp_name = args.exp_name

    print('=> Loading data ..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])

    if config.DATASET.TRAIN_DATASET == 'wildtrack':
        train_dataset = Wildtrack(config, transform, istrain=True)
        test_dataset = Wildtrack(config, transform, istrain=False)
    elif config.DATASET.TRAIN_DATASET == 'multiviewx':
        train_dataset = MultiviewX(config, transform, istrain=True)
        test_dataset = MultiviewX(config, transform, istrain=False)
    elif config.DATASET.TRAIN_DATASET == 'gmvd':
        train_dataset = GMVD(config, transform, istrain=True)
        test_dataset = GMVD(config, transform, istrain=False)

    sampler_train = torch.utils.data.RandomSampler(train_dataset)
    sampler_val = torch.utils.data.SequentialSampler(test_dataset)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.TRAIN.BATCH_SIZE, drop_last=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        num_workers=config.WORKERS,
        pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        sampler=sampler_val,
        pin_memory=True,
        num_workers=config.WORKERS)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = get_model(config, True, True, config.DATASET.IMAGE_SIZE)
    model.to(device)
    model_without_ddp = model

    optimizer = get_optimizer(model_without_ddp, args.weight_decay,
                              config.DECODER.optimizer)

    start_epoch = 0
    end_epoch = config.TRAIN.END_EPOCH

    if isinstance(config.DECODER.lr_decay_epoch, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.DECODER.lr_decay_epoch, gamma=0.1)
        print("Using MultiStepLR learning rate schedule")
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.DECODER.lr_decay_epoch, eta_min=1e-5)
        print("Using CosineAnnealingLR learning rate schedule")

    print('=> Training...')

    best_result = 350.0
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}/{}'.format(epoch, end_epoch))
        print('current lr {}'.format(optimizer.param_groups[0]["lr"]))

        model.train()
        losses = losses_ce = losses_bev = losses_bbox = losses_giou = losses_bbox_ce = 0

        for frame_id, (inputs, targets, meta) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in targets.items()}
            meta = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in meta.items()}

            output, loss, loss_dict = model(views=inputs, meta=meta,
                                            targets=targets, is_train=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()
            if 'loss_ce' in loss_dict:
                losses_ce += loss_dict['loss_ce'].item()
            if 'loss_bev' in loss_dict:
                losses_bev += loss_dict['loss_bev'].item()
            if 'loss_bbox' in loss_dict:
                losses_bbox += loss_dict['loss_bbox'].item()
            if 'loss_giou' in loss_dict:
                losses_giou += loss_dict['loss_giou'].item()
            if 'loss_bbox_ce' in loss_dict:
                losses_bbox_ce += loss_dict['loss_bbox_ce'].item()

            if (frame_id + 1) % 40 == 0 or frame_id + 1 == len(train_loader):
                n = frame_id + 1
                print(f'Train Epoch: {epoch}, Batch:{n}/{len(train_loader)}, '
                      f'loss: {losses / n:.6f}')
                print(f'        loss_bev_ce: {losses_ce / n:.6f}, '
                      f'loss_bev: {losses_bev / n:.6f}')
                print(f'        loss_img_ce: {losses_bbox_ce / n:.6f}, '
                      f'loss_bbox: {losses_bbox / n:.6f}, '
                      f'loss_giou: {losses_giou / n:.6f}')
                print()

        losses = losses / len(train_loader)
        print("=>>>>>>> Total Loss:", losses)

        lr_scheduler.step()

        # ---- Evaluation ----
        model.eval()
        worldgrid_size = config.DATASET.WORLDGRID
        moda_gt_list = []
        moda_pred_list = []
        max_objects = 100
        bev_threshold = 0.6
        nms_radius = 15

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

                worldgrid_size = meta['grid_shape']
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
                    ])

            gtRaw = np.array(moda_gt_list)
            detRaw = np.array(moda_pred_list)
            frames = np.unique(detRaw[:, 0]) if detRaw.size else np.zeros(0)
            frame_ctr = 0
            gt_flag = det_flag = True
            gtAllMatrix = detAllMatrix = 0

            if detRaw is None or detRaw.shape[0] == 0:
                MODP = MODA = recall = precision = 0
            else:
                for t in frames:
                    idx = np.where(gtRaw[:, 0] == t)[0]
                    idx_len = len(idx)
                    tmp = np.zeros((idx_len, 4))
                    tmp[:, 0] = frame_ctr
                    tmp[:, 1] = np.arange(idx_len)
                    tmp[:, 2] = gtRaw[idx, 1]
                    tmp[:, 3] = gtRaw[idx, 2]
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

                recall, precision, MODA, MODP = CLEAR_MOD_HUN(gtAllMatrix, detAllMatrix)

            all_metrics = recall + precision + MODA + MODP
            print("     Threshold:", bev_threshold, "Max objects:", max_objects,
                  "NMS radius:", nms_radius)
            print("==========================")
            print("Recall:", recall)
            print("Precision:", precision)
            print("MODA:", MODA)
            print("MODP:", MODP)
            print("Total metric:", all_metrics)
            print("==========================")

            if all_metrics > (best_result - 1.) and epoch >= 1:
                if all_metrics > best_result:
                    best_result = all_metrics
                    print('     $$$$$  ^        ^  $$$$$')
                    print('     $$$$$  |  BEST  |  $$$$$')
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        final_output_dir,
                        f'{config.DATASET.TRAIN_DATASET}_epoch_{epoch}.pth'))
            print()


if __name__ == '__main__':
    main()

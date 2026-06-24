from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import numpy as np
import cv2
import json
import xml.etree.ElementTree as ET
from operator import itemgetter

import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from utils import geom, basic, vox
from PIL import Image
import random 
import math


intrinsic_camera_matrix_filenames = ['intr_Camera1.xml', 'intr_Camera2.xml', 'intr_Camera3.xml', 'intr_Camera4.xml',
                                     'intr_Camera5.xml', 'intr_Camera6.xml']
extrinsic_camera_matrix_filenames = ['extr_Camera1.xml', 'extr_Camera2.xml', 'extr_Camera3.xml', 'extr_Camera4.xml',
                                     'extr_Camera5.xml', 'extr_Camera6.xml']

def random_bev_aug(grid_size, hflip=0.5, vflip=0.5):
    height = grid_size[1]
    width = grid_size[0]
    
    # flipping
    F = np.eye(4)
    hflip = np.random.rand() < hflip
    
    if hflip:
        F[0, 0] = -1
        F[0, 3] = width
        
    vflip = np.random.rand() < vflip
    if vflip:
        F[1, 1] = -1
        F[1, 3] = height
    
    return F


def random_affine(img, bboxs, pids, hflip=0.0, degrees=(-0, 0), translate=(.2, .2), scale=(0.8, 1.2), shear=(-0, 0),
                  borderValue=(128, 128, 128)):

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # flipping
    F = np.eye(3)
    hflip = np.random.rand() < hflip
    if hflip:
        F[0, 0] = -1
        F[0, 2] = width

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * width + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * height + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R @ F  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    n = bboxs.shape[0]
    area0 = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1])

    # warp points
    xy = np.ones((n * 4, 3))
    xy[:, :2] = bboxs[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(n, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

    # reject warped points outside of image
    np.clip(xy[:, 0], 0, width - 1, out=xy[:, 0])
    np.clip(xy[:, 2], 0, width - 1, out=xy[:, 2])
    np.clip(xy[:, 1], 0, height - 1, out=xy[:, 1])
    np.clip(xy[:, 3], 0, height - 1, out=xy[:, 3])
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    bboxs = xy[i]
    pids = pids[i]

    return imw, bboxs, pids, M, hflip

class MultiviewX(Dataset):
    def __init__(self, cfg, transform, istrain):
        self.__name__ = 'MultiviewX'
        self.root = cfg.DATASET.ROOT
        self.ori_shape = cfg.DATASET.ORIGINAL_SIZE
        self.img_shape = cfg.DATASET.IMAGE_SIZE
        self.worldgrid_shape = cfg.DATASET.WORLDGRID
        self.transform = transform
        
        self.num_cam = cfg.DATASET.CAMERA_NUM
        self.num_frame = cfg.DATASET.NUM_FRAME
                
        self.worldcoord_from_worldgrid_mat = np.array(      
            [[0.025,   0,   0, 0], 
             [0,   0.025,   0, 0], 
             [0,     0, 0.025,    0], 
             [0,     0,   0,    1]])
        
        self.intrinsic_matrices, self.extrinsic_matrices, self.distortion_coeffs = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        
        width, height = self.ori_shape
        self.optimal_K_matrices = []
        for cam in range(self.num_cam):
            if cam == 3:    # wrong intrinsic
                optimal_K, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic_matrices[0], 
                                                         self.distortion_coeffs[cam], 
                                                         (width, height), 1, (width, height))
            else:
                optimal_K, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic_matrices[cam], 
                                                            self.distortion_coeffs[cam], 
                                                            (width, height), 1, (width, height))
            self.optimal_K_matrices.append(optimal_K)
        
        self.istrain = istrain
        self.max_objects = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        
        if self.istrain:
            frame_range = range(0, int(self.num_frame * 0.9))
        else:
            frame_range = range(int(self.num_frame * 0.9), self.num_frame)  # WILDTRACK test on 40 last frames
        self.img_fpaths = self.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.transfer(frame_range)  # get annotation
        # self.img_gt => person boxes w/ IDs
        # self.world_gt => person positions w/ IDs

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.prepare_gt()   # create groundtruth BEV annotation

        self.calibration = {}
        self.cam_setting()
        
    def get_worldgrid_from_pos(self, pos):
        grid_x = pos % 1000
        grid_y = pos // 1000
        return np.array([grid_x, grid_y], dtype=int)
        
    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths
    
    def transfer(self, frame_range):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)  # dict_keys(['personID', 'positionID', 'views'])
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]

                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()   # 400239 => (833, 399)
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))  # (951, 346)
                    world_pids.append(pedestrian['personID'])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))
    
    def cam_setting(self):
        intrinsic = torch.tensor(np.stack(self.intrinsic_matrices, axis=0), dtype=torch.float32)  # S,3,3
        intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()  # S,4,4
        self.calibration['intrinsic'] = intrinsic
        
        self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        self.calibration['extrinsic'][:, :3] = torch.tensor(
            np.stack(self.extrinsic_matrices, axis=0), dtype=torch.float32)
    
    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')
    
    def undistort_points(self, points, cam):
        point_2d = cv2.undistortPoints(
            np.ascontiguousarray(np.float32(points)).reshape((1,-1,2)),
            self.intrinsic_matrices[cam],
            self.distortion_coeffs[cam],
            P=self.optimal_K_matrices[cam]
        ).squeeze(axis=1).reshape((len(points),4))
        return point_2d
    
    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                      intrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = fp_calibration.getNode('camera_matrix').mat()
        distortion_coeff =  fp_calibration.getNode('distortion_coefficients').mat().squeeze()
        fp_calibration.release()

        extrinsic_camera_path = os.path.join(self.root, 'calibrations', 'extrinsic')
        fp_calibration = cv2.FileStorage(os.path.join(extrinsic_camera_path,
                                                      extrinsic_camera_matrix_filenames[camera_i]),
                                         flags=cv2.FILE_STORAGE_READ)
        rvec, tvec = fp_calibration.getNode('rvec').mat().squeeze(), fp_calibration.getNode('tvec').mat().squeeze()
        fp_calibration.release()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix, distortion_coeff
    
    def get_image_data(self, index, cameras, aug_mat):
        frame = list(self.world_gt.keys())[index]
        
        imgs = []
        norm_img_pts_allcams = torch.ones((len(cameras), self.max_objects, 4), dtype=torch.float32)*100.
        norm_bev_pts = torch.ones((self.max_objects, 2), dtype=torch.float32)*100.
        norm_img_pids = torch.ones((len(cameras),self.max_objects)) * -1.
        norm_bev_pids = torch.ones(self.max_objects) * -1.
        
        bev_pts, bev_pids = self.world_gt[frame]
        if self.istrain:
            # warp points
            bev_homo = np.ones((len(bev_pts), 3))
            bev_homo[:,:2]=bev_pts
            bev_pts = (bev_homo @ aug_mat[:3, [0, 1, 3]].T)[:, :2]
        bev_pts = torch.tensor(bev_pts/np.array([self.worldgrid_shape[:2]]*len(bev_pts)))
        norm_bev_pts[0:len(bev_pts)] = bev_pts
        norm_bev_pids[0:len(bev_pids)] = torch.tensor(bev_pids)
        
        aff_Ms = []
        flips = []
        for cam in cameras:
            img = cv2.imread(self.img_fpaths[cam][frame])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            H, W, _  = img.shape # (1920, 1080)
            img_pts,img_pids = self.imgs_gt[frame][cam]
            
            if cam == 3:
                img = cv2.undistort(img, self.intrinsic_matrices[cam],
                                    self.distortion_coeffs[cam], None,
                                    self.optimal_K_matrices[cam])
            
            if cam == 3:
                img_pts = self.undistort_points(img_pts, cam)
            
            if self.istrain:
                img, img_pts, img_pids, aff_M, flip = random_affine(img, img_pts, img_pids)
                flips.append(flip)
            else:
                aff_M = np.eye(3)
                flips.append(False)
            aff_Ms.append(torch.from_numpy(aff_M).float())
            
            img_resized = cv2.resize(img, self.img_shape)   # (720, 1280, 3)
            imgs.append(F.to_tensor(img_resized))   # /255
            
            norm_img_pts = self.get_img_gt(img_pts, H, W)
            for idx in range(len(img_pids)):
                pid_pos = torch.where(norm_bev_pids==img_pids[idx])[0][0]
                norm_img_pts_allcams[cam][pid_pos] = norm_img_pts[idx]
                norm_img_pids[cam][pid_pos] = img_pids[idx]
        
        return torch.stack(imgs), norm_img_pts_allcams, norm_img_pids, \
                norm_bev_pts, norm_bev_pids, torch.stack(aff_Ms), flips
    
    def get_img_gt(self, img_pts, H, W):  # normalized cooridinates  (cx, cy, w, h)
        xmin = np.clip(img_pts[:, 0], 0, W)
        ymin = np.clip(img_pts[:, 1], 0, H)
        xmax = np.clip(img_pts[:, 2], 0, W)
        ymax = np.clip(img_pts[:, 3], 0, H)
        
        cxcywh = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin), axis=1)   # center pts of the box
        # Normalize
        cxcywh[:,[0,2]] = cxcywh[:,[0,2]] / W
        cxcywh[:,[1,3]] = cxcywh[:,[1,3]] / H

        norm_cxcywh = torch.tensor(cxcywh, dtype=torch.float32)
        
        return norm_cxcywh
    
    def __len__(self):
        return len(self.world_gt.keys())
    
    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]
        cameras = list(range(self.num_cam))  # TODO: cam dropout?
        
        aug_mat = np.eye(4)
        worldcoord_from_worldgrid = torch.tensor(self.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)
        
        # images
        imgs, img_pts, img_pids, bev_pts, bev_pids, aff_Ms, flips = self.get_image_data(index, cameras, aug_mat)
        num_cams = len(cameras)
        # aug_mat = ref_T_aug
        ref_T_global = worldgrid_T_worldcoord 
        
        pix_T_cams = self.calibration['intrinsic']  # 4x4
        affpix_T_cams_ = torch.matmul(aff_Ms,pix_T_cams[:, :3, :3])   # # 3x3
        affpix_T_cams = geom.merge_intrinsics(*geom.split_intrinsics(affpix_T_cams_)).squeeze() # 4x4
        
        cams_T_global = self.calibration['extrinsic']
        global_T_cams = torch.inverse(cams_T_global)
        ref_T_cams = torch.matmul(ref_T_global.repeat(num_cams, 1, 1), global_T_cams)
        cams_T_ref = torch.inverse(ref_T_cams)
        affpix_T_ref = torch.matmul(affpix_T_cams,cams_T_ref)
        affpix_T_ref_ = torch.matmul(affpix_T_cams[:, :3, :3],cams_T_ref[:, :3, [0, 1, 3]])  
        ref_T_affpix = torch.inverse(affpix_T_ref_)
        
        if self.transform:
            imgs = self.transform(imgs)      
        

        intrinsics = self.calibration['intrinsic']
        extrinsics = self.calibration['extrinsic']

        target = {
            # bev
            "bev_pts": bev_pts,
            "bev_pids": bev_pids,
            # img
            "img_pts": img_pts,  
            "img_pids": img_pids  
        }
        
        meta = {
            'intrinsic': intrinsics,  # S,4,4
            'extrinsic': extrinsics,  # S,4,4
            'optimal_K': self.optimal_K_matrices,
            'dist_coeff':self.distortion_coeffs,
            'ref_T_global': worldgrid_T_worldcoord,  # 4,4
            'bev_T_pix':ref_T_affpix,
            'pix_T_bev':affpix_T_ref,
            'cam_T_global':cams_T_global,
            'global_T_cam':global_T_cams,
            'num_cameras': self.num_cam,
            'grid_shape':self.worldgrid_shape,
            'root': self.root,
            'frame': frame,
            'flip': flips
        }
        
        if self.istrain:
            # Randomly drop 1-3 cameras
            num_cameras_to_drop = random.randint(0,  3)  # Ensure at least 1 camera remains
            cameras_to_drop = random.sample(range(self.num_cam), num_cameras_to_drop)
            
            for cid in cameras_to_drop:
                imgs[cid] = torch.zeros(3,self.img_shape[1],self.img_shape[0])
                target['img_pts'][cid] = torch.ones((self.max_objects, 4), dtype=torch.float32)*100
                target['img_pids'][cid] = torch.ones(self.max_objects) * -1.

        return imgs, target, meta

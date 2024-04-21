import argparse
import glob
import numpy as np
import os
import time
import torch.utils.data
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor
import json
import torch.nn.functional as F
import time
import cv2 as cv
import torch
from shapely.geometry import Polygon, Point

import argparse

from .model import Linear_net_small
from PIL import Image
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from .models_S.superpoint import SuperPoint
from .models_S.matching import Matching

from .models.matching import Matching
from .models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
default_config = {
        'descriptor_dim': 256,
        'nms_radius': 3,
        'keypoint_threshold': 0.05,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

class Matcher():
    def __init__(self, im_co, mask_points, \
                    path_m='./la_matcher/weights/model_new_temp40.pth', \
                    config=default_config, device=default_device) -> None:
        self.device = device
        # 模型初始化
        self.net = SuperPoint(config).to(device)
        self.f_net = Linear_net_small(4,256,8,256,256,4).to(device)
        self.params = torch.load(path_m, map_location= device)
        self.f_net.load_state_dict(torch.load(path_m, map_location=device))
        self.net.eval()
        self.f_net.eval()
        
        self.p_co, self.d_co = self.get_key_points(im_co)
        self.w, self.h = im_co.shape[1], im_co.shape[0]
        self.im_co = im_co
        for i in range(len(mask_points)):
            mask_points[i] = [mask_points[i][0] / self.w, mask_points[i][1] / self.h]
        self.mask_points = (np.array(mask_points)*512).astype(int)
        
    def get_key_points(self, im):
        # resize并转为tensor 
        im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (512,512))
        im = torch.from_numpy(im).float().to(self.device).float()/255.
        im = im.view(1,1,512,512)
        # 得到关键点
        out_p, desc = self.net(None, im)
        kp = out_p['keypoints']
        kp = np.array(kp[0].cpu().numpy()) # 方便后续变形
        p_int = kp.astype(int)
        batch_size=1
        p = torch.from_numpy(p_int).to(self.device).float().view(1,p_int.shape[0],2)
        
        out, desc = self.net(None, im, my_p=p)
        d = out['descriptors'].transpose(1,2).to(self.device) # 点的描述子
        # 返回关键点位置和关键点描述符
        return p, d
    
    def vis_superpoints(self, im):
        h, w = im.shape[0], im.shape[1]
        p, d = self.get_key_points(im)
        p = np.array(p).reshape(-1, 2)
        for i in range(p.shape[0]):
            px = int(p[i][0] * w / 512)
            py = int(p[i][1] * h / 512)
            cv.circle(im, (px, py), 5, (0, 255, 0), -1)
        return im
    
    def match_key_points(self, im_re):
        p1, d1 = self.p_co, self.d_co
        p2, d2 = self.get_key_points(im_re)
        p1_to_match, p2_to_match = np.array(p1.cpu().numpy().astype(int)).reshape(-1, 2), \
                                np.array(p2.cpu().numpy().astype(int)).reshape(-1, 2)
        # 把点的坐标变为LA_matcher的input shape
        p1 = (p1/256.).repeat([1,1,2])
        p2 = (p2/256.).repeat([1,1,2])
        out, point_map = self.f_net(None, d2, d1,  p2, p1, ocl= True)
        res = torch.argmax(point_map, dim=2).cpu().numpy() # 找到每个位置处特征值最大的索引, 即最佳匹配点
        res = res[0] # 变为一维索引
        
        # 其中的每个元素表示了第二张图像中的一个特征点匹配到第一张图像中的哪个特征点, 或者是否没有匹配到
        res[res==p2.shape[1]] = -1 # 去除无效点
        scores = torch.max(point_map, dim=2)[0]
        scores = scores.detach().cpu().numpy()[0]
        res[scores<0.4] = -1 # 去掉匹配度弱的点
        # 选择出那些成功匹配的特征点的坐标
        kp_successfully_matched_1 = p1_to_match[res>=0] # res中有值的点, 代表p1这个点匹配成功
        kp_successfully_matched_2 = p2_to_match[res[res>=0]] # p2匹配成功的点
        # 计算单应性矩阵, 得到单应性矩阵M和布尔矩阵mask(是否是内点)
        M, mask = cv2.findHomography(kp_successfully_matched_1.astype(np.float32).reshape(-1,1,2), \
                    kp_successfully_matched_2.astype(np.float32).reshape(-1,1,2), cv.RANSAC,6.0)
        mask = mask.reshape(-1)
        mask = mask==1 # 筛选出内点
        kp_successfully_matched_1 = kp_successfully_matched_1[mask]
        kp_successfully_matched_2 = kp_successfully_matched_2[mask]
        
        # 筛选出仅在im1区域内的匹配点
        im1_polygon = Polygon(self.mask_points)
        matched_points_in_im1 = []
        matched_points_in_im2 = []
        for i in range(len(kp_successfully_matched_1)):
            point1 = Point(kp_successfully_matched_1[i])
            point2 = Point(kp_successfully_matched_2[i])
            if im1_polygon.contains(point1):
                matched_points_in_im1.append(kp_successfully_matched_1[i])
                matched_points_in_im2.append(kp_successfully_matched_2[i])
        
        return np.array(matched_points_in_im1), np.array(matched_points_in_im2), M

    def vis_matched_points(self, im_re):
        kp_1, kp_2, _ = self.match_key_points(im_re)
        total_im = np.concatenate([cv2.resize(self.im_co, (512,512)), cv2.resize(im_re, (512,512))], 1)
        for i in range(len(kp_1)):
            pt1 = (int(kp_1[i][0]), int(kp_1[i][1]))
            pt2 = (int(kp_2[i][0] + 512), int(kp_2[i][1]))
            cv.line(total_im, pt1, pt2, (205, 0, 0), 1)
            cv.circle(total_im, pt1, 2, (0, 255, 0), -1)
            cv.circle(total_im, pt2, 2, (0, 255, 0), -1)
        return total_im
    

        
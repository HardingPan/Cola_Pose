import cv2
import numpy as np
import matplotlib.pyplot as plt

import random
import copy

class Sift():
    def __init__(self, img_co) -> None:
        # sift特征初始化
        self.siftDetector= cv2.xfeatures2d.SIFT_create()
        self.kp_co, self.des_co, self.kp_map_co = self.compute(img_co)
        self.kp_re, self.des_re, self.kp_map_re = self.kp_co, self.des_co, self.kp_map_co
        # 特征匹配初始化
        self.matcher = cv2.BFMatcher() # 匹配器
        
    def read(self, img_gray):
        self.kp_re, self.des_re, self.kp_map_re = self.compute(img_gray)
    
    # 计算特征点, 输出关键点、描述符、特征点图
    def compute(self, img_gray):
        # 得到关键点列表和对应于每个关键点的描述符列表。
        kp, des = self.siftDetector.detectAndCompute(img_gray, None)
        img_gray_copy = img_gray.copy()
        kp_map = cv2.drawKeypoints(img_gray, kp, img_gray_copy, \
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return kp, des, kp_map
    
    def match(self, img_real, threshold=0.5, k=2):
        self.read(img_real)
        matches = self.matcher.knnMatch(self.des_co, self.des_re, k)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < threshold*n.distance:
                good.append([m])
        matches = []
        for pair in good:
            matches.append(list(self.kp_co[pair[0].queryIdx].pt \
                                + self.kp_re[pair[0].trainIdx].pt))
        matches = np.array(matches)
        return matches
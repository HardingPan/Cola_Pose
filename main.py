import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

from la_matcher.la_matcher import Matcher
from estimaye_2d_rt import EstimateRT, K, dist_coeffs

def transform_point(pt_homogeneous, R, t):
    # pt_homogeneous = np.append(pt, 1)  # Convert to homogeneous coordinates
    transformed_pt_homogeneous = R @ pt_homogeneous[:2] + t
    # print(transformed_pt_homogeneous)
    return transformed_pt_homogeneous

im_co = cv2.imread("/Users/panding/Cola_Pose/data/co_1.bmp")
im_re = cv2.imread("/Users/panding/Cola_Pose/data/re_1.bmp")
# # 使用相机内参矫正图像
# im_co = cv2.undistort(im_co, K, dist_coeffs)
# im_re = cv2.undistort(im_re, K, dist_coeffs)
w, h = im_re.shape[1], im_re.shape[0]
print(f"图片已读取, 宽: {w}, 高: {h}")
# 蒙版坐标, 合作目标的重要信息
mask_points = [(1060, 525), (1060, 2000), (2030, 2000), (2030, 525)]
matcher = Matcher(im_co, mask_points)

kp_co, kp_re, M = matcher.match_key_points(im_re) # 得到一一对应的关键点
print(kp_co)
print(f"匹配点已得到")
cv2.imwrite("./res/match.png", matcher.vis_matched_points(im_re))

estimater = EstimateRT(kp_co, kp_re)
R1, t1 = estimater.recover_pose_by_E()

# # 计算本质矩阵
# E, mask = cv2.findEssentialMat(kp_co, kp_re, K, cv2.RANSAC, 0.999, 1.0)
# print(f"得到本质矩阵 \n {E}")

# _, R, t, _ = cv2.recoverPose(E, kp_co, kp_re, K)
# t = t.flatten()
# # 将旋转矩阵和平移向量组合成4x4的齐次变换矩阵
# R_t_matrix = np.eye(4, 4)
# R_t_matrix[:3, :3] = R
# R_t_matrix[:3, 3] = t
# print(f"得到RT矩阵 \n {R_t_matrix}")

# keypoints = [(1146, 674), (1190, 1964), (2048, 1942), (2006, 638)]
keypoints = kp_co
keypoints[:, 0] = keypoints[:, 0] * w/ 512
keypoints[:, 1] = keypoints[:, 1] * h/ 512
print(type(keypoints))
# keypoints = []


for i in range(len(keypoints)):
    # # 转换为齐次坐标
    # point_co_homogeneous = np.array([random.randint(1060, 2030), random.randint(525, 2000), 1, 1])
    point_co_homogeneous = np.array([keypoints[i][0], keypoints[i][1], 1, 1])
    point_re = transform_point(point_co_homogeneous, R1, t1)
    cv2.circle(im_co, (int(point_co_homogeneous[0]), int(point_co_homogeneous[1])), 5, (0, 0, 255), -1)
    cv2.circle(im_co, (int(point_re[0][0]), int(point_re[0][1])), 5, (0, 255, 0), -1)
    cv2.circle(im_re, (int(point_re[0][0]), int(point_re[0][1])), 5, (0, 255, 0), -1)
cv2.imwrite("/Users/panding/Cola_Pose/res/im_co.png", im_co)
cv2.imwrite("/Users/panding/Cola_Pose/res/im_re.png", im_re)

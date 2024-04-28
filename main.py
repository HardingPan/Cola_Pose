import cv2
import matplotlib.pyplot as plt
import numpy as np

from la_matcher.la_matcher import Matcher
import dlt

# 相机标定数据
K = np.zeros((3, 3), dtype=np.float64)
K[0, 0] = 5549.0097832871
K[0, 2] = 1526.98275689249
K[1, 1] = 5548.28944954669
K[1, 2] = 1072.66338519075
K[2, 2] = 1
dist_coeffs = np.float64([0.00363334166314795, 0.0961137299027789, 0, 0])
    
im_co = cv2.imread("/Users/panding/workspace/ws_3/data/1/co_1.bmp")
im_re = cv2.imread("/Users/panding/workspace/ws_3/data/1/re_5.bmp")
# 使用相机内参矫正图像
# im_co = cv2.undistort(im_co, K, distCoeffs)
# im_re = cv2.undistort(im_re, K, distCoeffs)
w, h = im_re.shape[1], im_re.shape[0]

# 蒙版坐标, 合作目标的重要信息
mask_points = [(1060, 525), (1060, 2000), (2030, 2000), (2030, 525)]
matcher = Matcher(im_co, mask_points)
kp_co, kp_re, M = matcher.match_key_points(im_re) # 得到一一对应的关键点
kp_co, kp_re = np.float64(kp_co), np.float64(kp_re)
# # 利用合作目标信息生成三维的世界坐标系
object_points = dlt.generate_3d_point_cloud(kp_co, mask_points)
print("############object_points###############")
print(object_points)
print("############kp_re###############")
print(kp_re)
_, rvecs, tvecs = cv2.solvePnP(object_points, kp_re, K, dist_coeffs)
rotation_matrix, _ = cv2.Rodrigues(rvecs)
rt_matrix = np.hstack((rotation_matrix, tvecs))
print(rt_matrix)
# 随机生成一组点, 世界坐标系的位置用于计算相机2D坐标, 相机坐标系的位置用于可视化显示
random_points_w, random_points_c = dlt.generate_random_points([(1060, 525), (1060, 2000), (2030, 2000), (2030, 525)], 1)
print("############random_points_w###############")
print(random_points_w)
projected_points = dlt.project_points_to_image(random_points_w, rotation_matrix, tvecs, K)
print("############projected_points###############")
print(projected_points)

# # 利用合作目标信息生成三维的世界坐标系
# object_points = dlt.generate_3d_point_cloud(kp_co, mask_points)
# # 通过dlt方法求解6D位姿
# rt_matrix = dlt.estimate_camera_matrix(object_points, kp_co)
# # print(rt_matrix)
# # 随机生成一组点, 世界坐标系的位置用于计算相机2D坐标, 相机坐标系的位置用于可视化显示
# random_points_w, random_points_c = dlt.generate_random_points([(1060, 525), (1060, 2000), (2030, 2000), (2030, 525)])
# print(random_points_w)
# project_points = dlt.project_points_to_image(random_points_w, rt_matrix, w, h)
# print(project_points)
# cv2.imwrite("./111.png", matcher.vis_matched_points(im_re))
# # co_points, re_points, M = matcher.match_key_points(p_re, p_co, d_re, d_co, mask_points)
# # pose_matrix = dlt.estimate_camera_matrix(co_points, re_points)
# # print(pose_matrix)

# # pw = np.array([[1590*512/h, 1082*512/w, 10]])
# # pc = dlt.project_points_to_image(pw, pose_matrix)
# # pt = (int(pc[0][0]*h/512), int(pc[0][1]*w/512))
# for i in range(len(random_points_c)):
#     pw = (int(random_points_c[i][0]), int(random_points_c[i][1]))
#     cv2.circle(im_co, pw, 10, (0, 255, 0), -1)
#     # pc = (int(project_points[i][0]), int(project_points[i][1]))
#     # cv2.circle(im_re, pc, 10, (0, 255, 0), -1)
# # im_final = np.concatenate((im_co, im_re), 1)
# # cv2.imwrite("./res.png", im_final)
# cv2.imwrite("./111.png", im_co)

# im_re = cv2.resize(im_re, (512,512))
# for i in range(len(project_points)):
#     pc = (int(project_points[i][0]), int(project_points[i][1]))
#     cv2.circle(im_re, pc, 2, (0, 255, 0), -1)
# cv2.imwrite("./222.png", im_re)
# # pw = np.array([[1590, 1082, 10]])
# # pc = dlt.project_points_to_image(pw, pose_matrix, w, h)
# # pt = (int(pc[0][0]), int(pc[0][1]))
# # cv2.circle(im_re, pt, 2, (0, 255, 0), -1)
# # cv2.imwrite("./res.png", im_re)


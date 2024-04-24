import cv2
import matplotlib.pyplot as plt
import numpy as np

from la_matcher.la_matcher import Matcher
import dlt

    
im_co = cv2.imread("data/co_1.bmp")
im_re = cv2.imread("data/re_1.bmp")
w, h = im_re.shape[1], im_re.shape[0]
mask_points = [(1068, 552), (1128, 2024), (2122, 2016), (2058, 524)]
matcher = Matcher(im_co, mask_points)
kp_co, kp_re, M = matcher.match_key_points(im_re)
im_total = matcher.vis_matched_points(im_re)
cv2.imwrite("./res.png", im_total)
depth = 300
rt_matrix = dlt.estimate_camera_matrix(kp_co, kp_re, depth)
print(rt_matrix)
random_points = dlt.generate_random_points([(1068, 552), (1128, 2024), (2122, 2016), (2058, 524)], depth)
project_points = dlt.project_points_to_image(random_points, rt_matrix, w, h)

# co_points, re_points, M = matcher.match_key_points(p_re, p_co, d_re, d_co, mask_points)
# pose_matrix = dlt.estimate_camera_matrix(co_points, re_points)
# print(pose_matrix)

# pw = np.array([[1590*512/h, 1082*512/w, 10]])
# pc = dlt.project_points_to_image(pw, pose_matrix)
# pt = (int(pc[0][0]*h/512), int(pc[0][1]*w/512))
for i in range(len(random_points)):
    pw = (int(random_points[i][0]), int(random_points[i][1]))
    cv2.circle(im_co, pw, 10, (0, 255, 0), -1)
    pc = (int(project_points[i][0]), int(project_points[i][1]))
    cv2.circle(im_re, pc, 10, (0, 255, 0), -1)
im_final = np.concatenate((im_co, im_re), 1)
cv2.imwrite("./res.png", im_final)

# pw = np.array([[1590, 1082, 10]])
# pc = dlt.project_points_to_image(pw, pose_matrix, w, h)
# pt = (int(pc[0][0]), int(pc[0][1]))
# cv2.circle(im_re, pt, 2, (0, 255, 0), -1)
# cv2.imwrite("./res.png", im_re)
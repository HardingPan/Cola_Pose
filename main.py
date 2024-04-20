import cv2
import matplotlib.pyplot as plt
import numpy as np

from la_matcher.la_matcher import Matcher

def plot_matches(matches, total_img):
    match_img = total_img.copy()
    offset = total_img.shape[1]/2
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(match_img).astype('uint8')) #　RGB is integer type
    
    ax.plot(matches[:, 0], matches[:, 1], 'xr')
    ax.plot(matches[:, 2] + offset, matches[:, 3], 'xr')
     
    ax.plot([matches[:, 0], matches[:, 2] + offset], [matches[:, 1], matches[:, 3]],
            'r', linewidth=0.1)

    plt.savefig('./res.png')
    
im_co = cv2.imread("data/co_1.bmp")
im_re = cv2.imread("data/re_1.bmp")

matcher = Matcher()
p_co, d_co = matcher.get_key_points(im_co)
p_re, d_re = matcher.get_key_points(im_re)
w, h = im_co.shape[0], im_co.shape[1]
mask_points = [(1068, 552), (1128, 2024), (2122, 2016), (2058, 524)]
for i in range(len(mask_points)):
    mask_points[i] = [mask_points[i][0] / h, mask_points[i][1] / w]
# print(type(p))
# im = matcher.vis_superpoints(im_co)
# cv2.imwrite("./res.png", im)
# my_p1, my_p2, vis = matcher.compute(im_co, im_re)
# cv2.imwrite("./res.png", vis)
# kp1, kp2 = matcher.match_key_points(p_re, p_co, d_re, d_co, [(1094, 594), (1162, 2002), (2072, 1980), (2010, 580)])
im = matcher.vis_matched_points(im_co, im_re, mask_points)
cv2.imwrite("./res.png", im)
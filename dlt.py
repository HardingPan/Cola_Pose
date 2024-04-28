import numpy as np
import random
import cv2

def generate_3d_point_cloud(points, mask_points):
    o_x, o_y = mask_points[0][0]*512/3072, mask_points[0][1]*512/2048
    h = (mask_points[1][1] - o_y)*512/2048
    w = (mask_points[3][0] - o_x)*512/3072
    # 创建一个与图像大小相同的二维数组来表示三维点云
    point_depth = np.zeros((points.shape[0], 1))
    for i in range(len(points)):
        points[i][0] = (points[i][0]-o_x)*85/512
        points[i][1] = (points[i][1]-o_y)*127/512
        point_depth[i] = 44  # 在内部，高度设为44
    points_3d = np.hstack((points, point_depth))
    return points_3d

def estimate_camera_matrix(object_points, re_points):
    # 构建A矩阵
    A = []
    for i in range(len(object_points)):
        X, Y, Z = object_points[i]
        x, y = re_points[i]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y])
    A = np.array(A)
    # 对A进行SVD分解
    _, _, V = np.linalg.svd(A)
    # 提取相机矩阵的最后一列（即V的最后一列）
    P = V[-1].reshape(3, 4)

    return P


# def project_points_to_image(points_3d, camera_matrix, w, h, target_w=512, target_h=512):
#     # 将3D点缩放到512x512的尺度
#     points_3d_scaled = points_3d.copy()
#     points_3d_scaled[:, 0] = points_3d_scaled[:, 0]*target_w / w
#     points_3d_scaled[:, 1] = points_3d_scaled[:, 1]*target_h / h
#     # 将3D点添加一个额外的维度，以便与矩阵相乘
#     points_3d_scaled = np.hstack((points_3d_scaled, np.ones((points_3d_scaled.shape[0], 1))))
#     print(points_3d_scaled)
#     # 将3D点投影到2D图像平面
#     points_2d = np.dot(camera_matrix, points_3d_scaled.T).T
#     # 将齐次坐标转换为欧几里得坐标
#     points_2d[:, 0] /= points_2d[:, 2]
#     points_2d[:, 1] /= points_2d[:, 2]
#     # 将2D坐标映射回原始尺度
#     points_2d[:, 0] *= w / target_w
#     points_2d[:, 1] *= h / target_h
    
#     return points_2d[:, :2]

# def project_points_to_image(world_points, rotation_matrix, translation_vector, camera_matrix):
#     # 将world_points表示为齐次向量形式
#     world_points_hom = np.hstack((world_points, np.ones((world_points.shape[0], 1))))
#     # 将齐次向量乘以旋转矩阵
#     print(rotation_matrix.shape, world_points_hom.T.shape)
#     camera_points_hom = np.dot(rotation_matrix, world_points_hom.T).T
#     # 将旋转后的齐次向量乘以平移向量
#     camera_points = camera_points_hom[:, :3] + translation_vector
#     # 归一化二维齐次坐标以得到实际的像素坐标
#     camera_points = camera_points / camera_points[:, 2:]
    
#     return camera_points

# def project_points_to_image(world_points, rotation_matrix, translation_vector, camera_matrix):
#     world_points_homogeneous = np.hstack((world_points, np.ones((len(world_points), 1))))
#     camera_points = rotation_matrix.dot(world_points_homogeneous.T).T + translation_vector
#     image_points, _ = cv2.projectPoints(camera_points, rotation_matrix, translation_vector, camera_matrix, np.zeros((1, 5)))
#     return image_points

def project_points_to_image(world_points, rotation_matrix, translation_vector, camera_matrix):
    rt_matrix = np.hstack((rotation_matrix, translation_vector))
    world_points_homogeneous = np.hstack((world_points, np.ones((len(world_points), 1))))
    camera_points_homogeneous = rt_matrix.dot(world_points_homogeneous.T)
    camera_points_normalized = camera_points_homogeneous / camera_points_homogeneous[:, -1:]
    print(camera_points_normalized)
    

def generate_random_points(mask_points, num_points=5):
    random_points_w = []
    random_points_c = []
    o_x, o_y = mask_points[0][0], mask_points[0][1]
    h = mask_points[1][1] - o_y
    w = mask_points[3][0] - o_x
    
    while len(random_points_w) < num_points:
        # 随机生成一个点在边界框内
        x_w = random.randint(0, 85)
        y_w = random.randint(0, 127)
        random_points_w.append((x_w, y_w, 44))

        x_c = x_w*w/85 + o_x
        y_c = y_w*h/127 + o_y
        random_points_c.append((x_c, y_c))
    return np.array(random_points_w).astype(float), np.array(random_points_c).astype(float)
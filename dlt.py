import numpy as np

def generate_3d_point_cloud(points):
    # 创建一个与图像大小相同的二维数组来表示三维点云
    point_depth = np.zeros((points.shape[0], 1))
    for i in range(len(points)):
        point_depth[i] = 10  # 在内部，高度设为10
    points_3d = np.hstack((points, point_depth))
    
    return points_3d

def estimate_camera_matrix(co_points, re_points):
    object_points = generate_3d_point_cloud(co_points)
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

def project_points_to_image(points_3d, camera_matrix):
    # 将3D点添加一个额外的维度，以便与矩阵相乘
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    # 将3D点投影到2D图像平面
    points_2d = np.dot(camera_matrix, points_3d.T).T
    # 将齐次坐标转换为欧几里得坐标
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    # 返回2D点的x和y坐标
    return points_2d[:, :2]
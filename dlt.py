import numpy as np
from shapely.geometry import Polygon, Point

def generate_3d_point_cloud(points, h):
    # 创建一个与图像大小相同的二维数组来表示三维点云
    point_depth = np.zeros((points.shape[0], 1))
    for i in range(len(points)):
        point_depth[i] = h  # 在内部，高度设为h
    points_3d = np.hstack((points, point_depth))
    
    return points_3d

def estimate_camera_matrix(co_points, re_points, h):
    object_points = generate_3d_point_cloud(co_points, h)
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

def project_points_to_image(points_3d, camera_matrix, w, h, target_w=512, target_h=512):
    # 将3D点缩放到512x512的尺度
    points_3d_scaled = points_3d.copy()
    points_3d_scaled[:, 0] *= target_w / w
    points_3d_scaled[:, 1] *= target_h / h
    # 将3D点添加一个额外的维度，以便与矩阵相乘
    points_3d_scaled = np.hstack((points_3d_scaled, np.ones((points_3d_scaled.shape[0], 1))))
    # 将3D点投影到2D图像平面
    points_2d = np.dot(camera_matrix, points_3d_scaled.T).T
    # 将齐次坐标转换为欧几里得坐标
    points_2d[:, 0] /= points_2d[:, 2]
    points_2d[:, 1] /= points_2d[:, 2]
    # 将2D坐标映射回原始尺度
    points_2d[:, 0] *= w / target_w
    points_2d[:, 1] *= h / target_h
    
    return points_2d[:, :2]

def generate_random_points(vertices, h, num_points=5):
    polygon = Polygon(vertices)  # 指定使用 (x, y) 坐标顺序
    # 计算四边形的边界框
    x_min = min(v[0] for v in vertices)
    x_max = max(v[0] for v in vertices)
    y_min = min(v[1] for v in vertices)
    y_max = max(v[1] for v in vertices)
    random_points = []  # 生成随机点
    while len(random_points) < num_points:
        # 随机生成一个点在边界框内
        x = np.random.uniform() * (x_max - x_min) + x_min
        y = np.random.uniform() * (y_max - y_min) + y_min
        point = Point(x, y)
        # 判断该点是否在多边形内部
        if polygon.contains(point):
            random_points.append((x, y, h))
    return np.array(random_points)
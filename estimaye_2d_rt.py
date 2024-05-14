import cv2
import matplotlib.pyplot as plt
import numpy as np

from la_matcher.la_matcher import Matcher

# 相机标定数据
K = np.zeros((3, 3), dtype=np.float64)
K[0, 0] = 5549.0097832871
K[0, 2] = 1526.98275689249
K[1, 1] = 5548.28944954669
K[1, 2] = 1072.66338519075
K[2, 2] = 1
dist_coeffs = np.float64([0.00363334166314795, 0.0961137299027789, 0, 0])

class EstimateRT():
    def __init__(self, kp_1, kp_2) -> None:
        self.kp_1_normalize = self.normalize_to_unit_variance(kp_1)
        self.kp_2_normalize = self.normalize_to_unit_variance(kp_2)
        self.kp_1 = self.remap_points(kp_1)
        self.kp_2 = self.remap_points(kp_2)
        
    def normalize_to_unit_variance(self, points):
        # 计算均值
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])
        # 计算方差，并添加一个小的常数以避免除以零
        var_x = np.var(points[:, 0]) + 1e-10
        var_y = np.var(points[:, 1]) + 1e-10
        # 标准化坐标
        normalized_points = points.copy()
        normalized_points[:, 0] = (points[:, 0] - mean_x) / np.sqrt(var_x)
        normalized_points[:, 1] = (points[:, 1] - mean_y) / np.sqrt(var_y)
        return normalized_points
    
    def remap_points(self, points):
        points = points.astype(np.float32)
        scale_x = 3072 / 512
        scale_y = 2048 / 512
        points[:, 0] = points[:, 0] * scale_x
        points[:, 1] = points[:, 1] * scale_y
        points = cv2.undistortPoints(np.expand_dims(points, axis=1), K, None)
        return points
    
    def eight_point_algorithm(self):
        """用于计算基本矩阵"""
        A = np.zeros((self.kp_1.shape[0], 9))
        # 构建矩阵A以用于线性方程组
        for i in range(self.kp_1.shape[0]):
            x1, y1 = self.kp_1[i]
            x2, y2 = self.kp_2[i]
            A[i, 0:3] = x1*x2, y1*x2, x2
            A[i, 3:6] = x1, y1, 1
            A[i, 6:9] = -(x2*y1), -(y1*y2), -y2
        # 使用最小二乘法求解线性方程组
        U, S, Vh = np.linalg.svd(A)
        F = Vh[-1, :].reshape(3, 3)
        # 由于F矩阵的秩为2或更少，我们可以通过奇异值分解得到两个可能的解
        F = F / F[2, 2]  # 归一化F矩阵的最后一行
        # 检查F矩阵的秩，如果秩小于2，需要重新计算
        if np.linalg.matrix_rank(F) < 2:
            raise ValueError("The matrix F has rank less than 2.")
        # 由于F矩阵可能存在符号错误，我们检查行列式并相应地调整符号
        if np.linalg.det(F) < 0:
            F[:, 2] = -F[:, 2]
        return F
    
    def decompose_F_matrix(self):
        """使用基本矩阵求解rt"""
        F = self.eight_point_algorithm()
        U, S, Vt = np.linalg.svd(F)
        R = Vt[0:2, 0:2].T  # 旋转矩阵
        t = Vt[0:2, 2]      # 平移向量
        # 由于旋转矩阵的行列式应该是1，需要检查并可能调整符号
        if np.linalg.det(R) < 0:
            R = -R
            t = -t
        return R, t.flatten()
    
    def recover_pose_by_E(self):
        E, mask = cv2.findEssentialMat(self.kp_1, self.kp_2, K)
        _, R, t, mask = cv2.recoverPose(E, self.kp_1, self.kp_2, K)
        R_2d = R[:2, :2]
        t_2D = t[:2]
        return R_2d, t_2D
        # # 奇异值分解
        # U, S, Vt = np.linalg.svd(E)
        # # 恢复旋转矩阵 R 和平移向量 t
        # R1 = U.dot(Vt)  # R1 是一个3x3的矩阵
        # t1 = Vt[:, 2]
        # # 处理符号歧义，检查 R1 的行列式
        # if np.linalg.det(R1) < 0:
        #     R1 = -R1
        #     t1 = -t1
        # # 另一种可能的解
        # R2 = U.dot(np.diag([-1, -1, 1])).dot(Vt)  # 改变 Vt 的前两个奇异值的符号
        # t2 = np.array([-t1[0], -t1[1], t1[2]])
        # # 同样检查 R2 的行列式
        # if np.linalg.det(R2) < 0:
        #     R2 = -R2
        #     t2 = -t2
        # return R1, t1, R2, t2
    
        

    
        

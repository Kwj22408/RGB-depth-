import numpy as np
import cv2
import open3d as o3d


def create_color_point_cloud(depth_path, rgb_path):
    """
    利用深度图、RGB图和相机内参生成彩色点云
    :param depth_path: 对齐到RGB的深度图路径（建议为16位深度图，单位：毫米）
    :param rgb_path: RGB图像路径
    :return: 彩色点云对象
    """
    # ========== 1. 定义相机内参和畸变参数 ==========
    # 内参参数
    width = 1920
    height = 1080
    fx = 1123.51  # x轴焦距
    fy = 1122.9  # y轴焦距
    cx = 957.35  # x轴主点坐标
    cy = 532.874  # y轴主点坐标

    # 畸变参数（OpenCV顺序：k1,k2,p1,p2,k3,k4,k5,k6）
    k1 = 0.0793481
    k2 = -0.106039
    p1 = 3.25515e-05
    p2 = 0.000350022
    k3 = 0.0428345
    k4 = 0.0
    k5 = 0.0
    k6 = 0.0
    distortion_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

    # 构建内参矩阵K
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    # ========== 2. 读取图像数据 ==========
    # 读取深度图（16位，单位：毫米）
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # 读取RGB图并转换为RGB格式（OpenCV默认BGR）
    rgb_img = cv2.imread(rgb_path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    # 校验图像尺寸
    if depth_img.shape != (height, width) or rgb_img.shape[:2] != (height, width):
        raise ValueError(f"图像尺寸不匹配！要求1920x1080，深度图尺寸：{depth_img.shape}，RGB图尺寸：{rgb_img.shape}")

    # ========== 3. 生成点云数据 ==========
    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.astype(np.float64)
    v = v.astype(np.float64)

    # 深度值转换为米（如果深度图单位是毫米）
    z = depth_img.astype(np.float64) / 1000.0
    # 过滤无效深度点（深度值为0或负数）
    valid_mask = z > 0

    # 计算3D坐标
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 提取有效点的坐标和颜色
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_z = z[valid_mask]
    valid_colors = rgb_img[valid_mask] / 255.0  # Open3D要求颜色值在[0,1]区间

    # 构建点云
    point_cloud = o3d.geometry.PointCloud()
    # 设置点坐标
    points = np.column_stack((valid_x, valid_y, valid_z))
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # 设置点颜色
    point_cloud.colors = o3d.utility.Vector3dVector(valid_colors)

    return point_cloud


# ========== 主函数 ==========
if __name__ == "__main__":
    # 替换为你的深度图和RGB图路径
    DEPTH_PATH = r"D:\pigtest\origin\07.png"  # 对齐到RGB的深度图
    RGB_PATH = r"D:\pigtest\RGB\07.png"  # RGB图像

    try:
        # 生成彩色点云
        color_pcd = create_color_point_cloud(DEPTH_PATH, RGB_PATH)

        # 可选：去除离群点（优化点云质量）
        color_pcd, _ = color_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # 可视化点云
        o3d.visualization.draw_geometries([color_pcd], window_name="彩色点云")

        # 保存点云（可选，保存为ply格式）
        o3d.io.write_point_cloud("7viewscloud007.ply", color_pcd)
        print("彩色点云已保存为 color_point_cloud.ply")

    except Exception as e:
        print(f"生成点云失败：{e}")
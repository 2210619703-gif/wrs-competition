import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from ur3e.ur3e import UR3e
from manipulator_interface import ManipulatorInterface
from wrs.modeling.geometric_model import gen_pointcloud
from scipy.spatial import ConvexHull
import pyvista as pv

def compute_tcp_positions(pkl_folder, max_points=100000):
    arm = UR3e(enable_cc=True)
    tcp_positions = []
    manipulabilities = []
    total_tcp_count = 0

    for filename in os.listdir(pkl_folder):
        if filename.endswith('.pkl'):
            file_path = os.path.join(pkl_folder, filename)
            with open(file_path, 'rb') as f:
                joint_angles_list = pickle.load(f)

                for joint_angles in joint_angles_list:
                    if len(tcp_positions) >= max_points:
                        break
                    if joint_angles.shape[0] == 5:
                        joint_angles = np.append(joint_angles, 0)
                    tcp_pos, rotation_matrix = arm.fk(joint_angles, update=True)
                    manipulability = arm.manipulability_val()

                    if not np.isnan(manipulability):
                        tcp_positions.append(tcp_pos)
                        manipulabilities.append(manipulability)
                        total_tcp_count += 1

    tcp_positions = np.array(tcp_positions)
    manipulabilities = np.array(manipulabilities)

    if manipulabilities.size > 0:
        print(f"可操作性范围: {np.min(manipulabilities)}, {np.max(manipulabilities)}")
    else:
        print("No valid manipulability values found.")
    print(f"TCP Positions 数量: {total_tcp_count}")

    if total_tcp_count > 0:
        # 筛选可操作性大于 0.005 的点
        high_manipulability_points = tcp_positions[manipulabilities > 0.005]

        if len(high_manipulability_points) > 3:  # 至少要有4个点才能构成三维凸包
            # 生成并绘制凸包
            voxelized_cloud = plot_convex_hull(high_manipulability_points)

            # 可视化体素化结果
            if voxelized_cloud is not None:
                p = pv.Plotter()
                p.add_mesh(voxelized_cloud, show_edges=True, color='blue', opacity=0.5)
                p.show()


def plot_convex_hull(filtered_positions):
    hull = ConvexHull(filtered_positions)

    # 计算凸包的点
    hull_points = filtered_positions[hull.vertices]

    # 创建 PolyData 对象
    cloud = pv.PolyData(hull_points)

    # 使用 Delaunay 3D 方法创建网格
    delaunay = cloud.delaunay_3d()

    # 体素化
    voxel_size = 0.01  # 设置体素大小
    voxelized_cloud = pv.voxelize(delaunay, voxel_size)

    return voxelized_cloud

if __name__ == '__main__':
    compute_tcp_positions('data/fk11')
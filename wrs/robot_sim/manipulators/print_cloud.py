import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from ur3e.ur3e import UR3e
from manipulator_interface import ManipulatorInterface
from wrs.modeling.geometric_model import gen_pointcloud
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R  # 用于从旋转矩阵提取欧拉角


def compute_tcp_positions(pkl_folder, max_points=1000000, base=None):
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

                        # 打印每个 TCP 位置的 x, y, z, rx, ry, rz
                        # 位置是 tcp_pos, 旋转矩阵是 rotation_matrix
                        x, y, z = tcp_pos
                        r = R.from_matrix(rotation_matrix)  # 创建旋转对象
                        rx, ry, rz = r.as_euler('xyz', degrees=False)  # 获取欧拉角 (单位是弧度)
                        print(f"TCP Position (x, y, z): ({x}, {y}, {z})")
                        print(f"Rotation (rx, ry, rz): ({rx}, {ry}, {rz})")
                        print("-" * 40)

    tcp_positions = np.array(tcp_positions)
    manipulabilities = np.array(manipulabilities)

    if manipulabilities.size > 0:
        print(f"Overall Manipulability range: {np.min(manipulabilities)}, {np.max(manipulabilities)}")
    else:
        print("No valid manipulability values found.")
    print(f"Total TCP Positions Count: {total_tcp_count}")

    if total_tcp_count > 0 and base is not None:
        cmap = plt.cm.plasma
        norm = BoundaryNorm(
            boundaries=np.linspace(0, np.max(manipulabilities), int(np.ceil(np.max(manipulabilities) / 0.005)) + 1),
            ncolors=cmap.N, clip=True)
        colors = cmap(norm(manipulabilities))

        rgba = np.zeros((total_tcp_count, 4))
        rgba[:, :3] = colors[:, :3]
        rgba[:, 3] = 1

        pointcloud_model = gen_pointcloud(tcp_positions, rgba=rgba, point_size=0.001)
        pointcloud_model.attach_to(base)

        # 筛选可操作性大于 0.015 的点
        high_manipulability_points = tcp_positions[manipulabilities > 0.015]

        if len(high_manipulability_points) > 3:  # 至少要有4个点才能构成三维凸包
            # 生成并绘制凸包
            plot_convex_hull(high_manipulability_points)


def plot_convex_hull(filtered_positions):
    hull = ConvexHull(filtered_positions)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2], color='blue',
               alpha=0.5, s=1)

    # Plot the convex hull edges
    for simplex in hull.simplices:
        ax.plot(filtered_positions[simplex, 0], filtered_positions[simplex, 1], filtered_positions[simplex, 2],
                color='orange')

    ax.set_title('Convex Hull of High Manipulability TCP Positions (>0.015)')
    ax.set_xlabel('X (dm)')
    ax.set_ylabel('Y (dm)')
    ax.set_zlabel('Z (dm)')
    plt.show()


if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    arm = UR3e(enable_cc=True)
    arm.fk(arm.rand_conf(), update=True)
    arm_mesh = arm.gen_meshmodel(alpha=1)
    arm_mesh.attach_to(base)

    compute_tcp_positions('data/fk11', base=base)
    base.run()

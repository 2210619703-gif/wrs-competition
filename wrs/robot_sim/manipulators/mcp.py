from ortools.linear_solver import pywraplp
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from ur3e.ur3e import UR3e

def compute_tcp_positions(pkl_folder, max_points=10000):
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

    if total_tcp_count > 0:
        high_manipulability_points = tcp_positions[manipulabilities > 0.005]

        if len(high_manipulability_points) > 3:
            hull = ConvexHull(high_manipulability_points)
            hull_center = np.mean(high_manipulability_points[hull.vertices], axis=0)
            hull_radius = np.max(np.linalg.norm(high_manipulability_points[hull.vertices] - hull_center, axis=1))

            print(f"凸包的最大半径: {hull_radius:.2f} meters")
            return hull_radius, hull_center

    print("No valid TCP positions found.")
    return None, None

def cover_points_with_ortools(apple_positions, sphere_radius):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver not available.")
        return

    num_apples = apple_positions.shape[0]
    sphere_centers = []
    selected_spheres = []

    # 定义球体的中心点，确保下平面与 xy 平面相切
    for i in range(num_apples):
        center = apple_positions[i]
        # 球体的 z 坐标是半径 r，确保球体下平面与 xy 平面相切
        sphere_centers.append((center[0], center[1], sphere_radius))
        selected_spheres.append(solver.BoolVar(f'sphere_{i}'))

    # 约束：每个苹果点至少被一个球体覆盖
    for i in range(num_apples):
        apple = apple_positions[i]
        coverage_constraints = []

        for j, center in enumerate(sphere_centers):
            # 计算苹果点与球体中心在 xy 平面上的距离
            distance_xy = np.linalg.norm(np.array((apple[0], apple[1])) - np.array((center[0], center[1])))
            # 检查是否在覆盖半径内
            if distance_xy <= sphere_radius:
                coverage_constraints.append(selected_spheres[j])

        # 添加覆盖约束，确保每个点被至少一个球体覆盖
        solver.Add(solver.Sum(coverage_constraints) >= 1)

    # 目标：最小化使用的球体数量
    objective = solver.Sum(selected_spheres)
    solver.Minimize(objective)

    # 求解问题
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print('最优解已被找到.')
        chosen_centers = [sphere_centers[i] for i in range(num_apples) if selected_spheres[i].solution_value() > 0.5]
        print(f'使用的球体数量: {len(chosen_centers)}')
        return chosen_centers, apple_positions
    else:
        print('没找到最优解.')
        return None, apple_positions


def solve_tsp(points):
    manager = pywrapcp.RoutingIndexManager(len(points), 1, 1)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        distance = np.linalg.norm(points[from_node] - points[to_node])
        print(f"Distance from {points[from_node]} to {points[to_node]} is {distance}")
        return distance

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        plan_output = 'Route:\n'
        route_distance = 0
        index = routing.Start(0)
        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            plan_output += f'{points[manager.IndexToNode(index)]} -> {points[manager.IndexToNode(next_index)]} ||| '
            route_distance += routing.GetArcCostForVehicle(index, next_index, 0)
            index = next_index
        plan_output += f'{points[manager.IndexToNode(index)]}\n'
        print(plan_output)
        return route_distance
    else:
        print("TSP未找到解")
        return None

if __name__ == '__main__':
    apple_positions = np.array([
        (1.19, 0.06, 0.6), (1.24, 0.2, 1.71), (1.66, 0.38, 0.91), (1.08, 0.69, 2.23), (0.97, 1.63, 2.25), (1.21, 1.52, 1.44), (1.77, 1.83, 1.27), (1.1, 1.92, 2.91), (1.69, 1.26, 2.12), (0.6, 1.31, 1.13), (1.02, 0.66, 1.83), (1.16, 1.27, 0.54), (1.06, 0.2, 2.66), (1.62, 0.18, 0.63), (1.29, 1.19, 2.5), (1.42, 1.11, 1.73), (0.77, 0.62, 0.58), (1.87, 0.61, 0.71), (1.31, 1.49, 2.48), (0.11, 0.59, 1.54), (0.76, 1.32, 1.05), (0.93, 0.94, 1.58), (0.71, 0.56, 2.08), (1.62, 0.13, 1.8), (1.34, 0.01, 2.86), (0.79, 0.18, 0.51), (1.34, 1.12, 0.21), (0.79, 1.85, 0.54), (0.1, 1.92, 0.55), (1.54, 1.7, 1.4), (1.67, 0.14, 0.14), (0.18, 0.93, 1.98), (1.98, 0.23, 0.38), (1.34, 0.07, 2.09), (0.75, 0.32, 0.4), (1.03, 1.44, 1.86), (1.31, 1.74, 1.6), (1.63, 1.05, 1.23), (0.65, 0.08, 0.91), (0.85, 1.88, 0.27), (1.28, 1.7, 1.85), (0.66, 1.2, 2.34), (0.27, 1.41, 0.9), (0.48, 1.58, 0.92), (1.12, 0.22, 0.69), (1.08, 0.54, 0.98), (1.78, 1.55, 2.19), (1.46, 0.98, 1.62), (1.6, 1.01, 0.46), (0.45, 0.69, 2.54), (1.96, 0.32, 2.65), (1.52, 0.11, 2.72), (1.34, 1.87, 2.2), (1.46, 0.65, 0.46), (1.26, 1.0, 1.12), (1.36, 1.86, 0.78), (1.24, 1.21, 0.39), (1.84, 1.53, 2.54), (1.42, 0.26, 1.98), (0.45, 0.94, 0.69), (0.77, 0.87, 2.41), (0.74, 0.7, 1.19), (1.49, 1.59, 2.43), (0.27, 0.05, 1.75), (1.83, 1.49, 0.6), (1.55, 0.25, 1.08), (1.69, 1.78, 2.38), (1.46, 0.13, 1.33), (1.11, 0.42, 1.7), (1.44, 0.08, 0.41), (0.76, 0.37, 2.35), (1.45, 0.84, 2.31), (0.12, 0.07, 1.25), (1.42, 0.05, 1.86), (0.87, 1.56, 0.54), (0.98, 1.44, 1.83), (1.56, 0.29, 0.15), (1.14, 1.17, 0.61), (0.14, 1.58, 1.76), (0.55, 0.28, 2.26), (0.22, 0.97, 0.07), (0.23, 0.15, 2.32), (0.59, 0.9, 1.91), (0.88, 1.9, 1.58), (0.9, 0.59, 1.08), (1.37, 1.63, 2.99), (0.24, 1.22, 0.85), (0.5, 2.0, 1.0), (0.15, 1.14, 0.43), (1.86, 0.27, 1.59), (0.41, 0.76, 1.58), (1.67, 1.89, 2.28), (1.46, 0.87, 2.61), (0.13, 0.2, 1.44), (0.32, 1.39, 1.79), (1.02, 0.66, 2.4), (0.9, 1.6, 0.57), (0.1, 0.47, 2.78), (1.99, 0.77, 1.23), (0.9, 1.73, 1.39)
    ])

    pkl_folder = 'data/fk11'
    radius, center = compute_tcp_positions(pkl_folder)

    if radius is not None and center is not None:
        chosen_centers, all_points = cover_points_with_ortools(apple_positions, radius)

        if chosen_centers:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")
            ax.set_title("3D Space with Apple Positions and Coverage Spheres (Optimal)")

            # 绘制覆盖的球体
            for center in chosen_centers:
                u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:25j]  # 增加分辨率
                x = radius * np.cos(u) * np.sin(v) + center[0]
                y = radius * np.sin(u) * np.sin(v) + center[1]
                z = radius * np.cos(v) + center[2]
                ax.plot_surface(x, y, z, color='blue', alpha=0.1, edgecolor='k')

            # 绘制苹果的位置
            covered_points = []
            uncovered_points = []

            for point in all_points:
                # 判断该点是否被覆盖
                if any(np.linalg.norm(point - center) <= radius for center in chosen_centers) and point[2] <= radius:
                    covered_points.append(point)
                else:
                    uncovered_points.append(point)

            covered_points = np.array(covered_points)
            uncovered_points = np.array(uncovered_points)

            # 绘制被覆盖的点（颜色为红色）
            ax.scatter(covered_points[:, 0], covered_points[:, 1], covered_points[:, 2], color='red', s=20,
                       label='Covered Apples')
            # 绘制未被覆盖的点（颜色为绿色）
            ax.scatter(uncovered_points[:, 0], uncovered_points[:, 1], uncovered_points[:, 2], color='green', s=20,
                       label='Uncovered Apples')

            # 绘制球心（颜色为蓝色）
            for center in chosen_centers:
                ax.scatter(center[0], center[1], center[2], color='blue', s=100, label='Sphere Centers')

            ax.legend()
            plt.show()

            # 计算每个球体覆盖点集的最短路径
            for center in chosen_centers:
                covered_points_in_sphere = [p for p in covered_points if np.linalg.norm(p - center) <= radius]
                if covered_points_in_sphere:
                    solve_tsp(covered_points_in_sphere)
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

def cover_points_with_ortools(apple_positions, sphere_radius, hull_radius):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver not available.")
        return

    num_apples = apple_positions.shape[0]
    sphere_centers = []
    selected_spheres = []

    for i in range(num_apples):
        center = apple_positions[i]
        sphere_centers.append((center[0], center[1], sphere_radius))  # Ensure the sphere center is just above the XY plane
        selected_spheres.append(solver.BoolVar(f'sphere_{i}'))

    for i in range(num_apples):
        apple = apple_positions[i]
        coverage_constraints = []

        for j, center in enumerate(sphere_centers):
            distance = np.linalg.norm(np.array((apple[0], apple[1])) - np.array((center[0], center[1])))
            if distance <= sphere_radius:
                coverage_constraints.append(selected_spheres[j])

        solver.Add(solver.Sum(coverage_constraints) >= 1)

    objective = solver.Sum(selected_spheres)
    solver.Minimize(objective)

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
        print(f'Total distance of the route: {route_distance:.2f} meters')
        return route_distance
    else:
        print("TSP未找到解")
        return None

if __name__ == '__main__':
    apple_positions = np.array([
        (1.53, 0.29, 2.18), (0.28, 0.46, 2.21), (1.95, 1.76, 0.19), (1.74, 1.25, 1.96), (1.31, 0.5, 0.2),
        (1.73, 1.25, 0.65), (0.91, 1.13, 1.88), (1.49, 0.04, 1.16), (1.32, 1.89, 2.03), (1.75, 1.95, 1.56),
        (0.73, 0.22, 2.83), (1.21, 1.1, 1.91), (1.62, 0.79, 0.11), (1.23, 1.61, 0.09), (0.73, 1.38, 1.53),
        (1.9, 1.72, 2.48), (0.81, 0.49, 1.21), (1.12, 1.74, 1.96), (0.14, 1.83, 1.0), (0.82, 0.37, 2.05),
        (0.18, 0.4, 0.17), (0.04, 1.85, 2.75), (1.68, 0.1, 2.91), (0.74, 0.09, 1.43), (0.05, 1.82, 0.77),
        (0.68, 0.98, 1.98), (0.19, 1.5, 2.33), (1.75, 1.97, 2.14), (1.58, 0.23, 1.91), (0.79, 0.66, 1.43),
        (1.3, 1.05, 0.05), (1.05, 1.13, 2.28), (0.15, 1.48, 2.51), (0.33, 0.1, 1.01), (1.28, 0.16, 1.09),
        (1.25, 0.06, 0.02), (0.87, 1.15, 2.93), (1.31, 1.63, 1.16), (1.85, 0.31, 0.23), (0.59, 1.96, 2.04),
        (1.26, 0.23, 0.19), (1.67, 1.26, 2.73), (0.55, 0.42, 0.16), (0.86, 1.76, 0.52), (1.71, 1.39, 0.05),
        (0.67, 0.32, 2.32), (1.76, 0.96, 0.93), (1.54, 0.41, 0.07), (1.84, 0.52, 0.49), (0.62, 0.67, 0.85),
        (0.37, 1.86, 2.61), (0.17, 1.63, 2.89), (1.91, 1.12, 2.27), (0.23, 1.0, 2.92), (1.95, 1.07, 2.18),
        (1.48, 1.74, 0.67), (1.15, 0.79, 2.78), (1.36, 0.47, 1.48), (0.91, 0.63, 1.25), (1.75, 0.75, 0.21),
        (0.64, 1.28, 1.76), (0.83, 0.81, 0.19), (0.61, 0.84, 0.14), (0.99, 1.25, 1.52), (0.87, 0.41, 0.67),
        (0.61, 1.58, 2.78), (1.03, 0.95, 1.77), (0.23, 0.53, 2.74), (0.13, 0.94, 1.43), (1.77, 1.69, 2.44),
        (0.22, 1.49, 0.8), (0.98, 1.73, 2.15), (1.78, 1.74, 0.05), (0.17, 0.75, 2.06), (0.75, 0.91, 2.32),
        (0.69, 0.06, 1.42), (0.94, 1.61, 1.23), (1.21, 0.44, 0.17), (0.86, 1.78, 0.32), (1.07, 1.99, 0.96),
        (1.13, 0.67, 1.51), (1.37, 1.83, 1.82), (1.18, 1.95, 1.42), (1.48, 1.54, 2.09), (0.76, 0.13, 2.28),
        (0.67, 0.68, 2.87), (0.86, 1.55, 1.16), (1.76, 0.15, 0.26), (1.33, 1.93, 1.42), (0.19, 1.35, 1.18),
        (1.7, 1.43, 0.43), (0.26, 0.75, 0.05), (0.4, 1.08, 0.77), (0.69, 0.8, 2.22), (0.01, 2.0, 0.75),
        (1.29, 0.88, 1.73), (0.75, 0.44, 0.85), (1.33, 0.03, 1.08), (0.04, 1.92, 1.47), (1.18, 0.2, 2.78),
        (0.15, 1.51, 2.64), (0.85, 1.06, 1.59), (1.33, 1.68, 2.9), (0.38, 1.08, 0.83), (0.15, 0.02, 1.92),
        (0.3, 0.15, 2.46), (1.75, 0.93, 1.18), (1.38, 1.36, 1.96), (1.31, 0.55, 0.41), (0.98, 1.91, 0.58),
        (0.47, 1.27, 1.6), (0.83, 1.67, 2.07), (0.67, 1.87, 2.8), (0.49, 0.82, 1.88), (1.38, 0.29, 1.94),
        (1.03, 1.89, 1.62), (1.36, 0.18, 2.0), (0.39, 1.81, 2.86), (0.54, 1.79, 1.25), (1.03, 1.51, 1.48),
        (0.01, 1.03, 0.43), (1.54, 0.63, 0.93), (1.64, 0.92, 1.62), (0.49, 1.72, 2.49), (0.63, 1.01, 3.0),
        (1.8, 0.08, 1.04), (0.85, 1.72, 0.72), (0.74, 0.34, 2.46), (1.7, 0.18, 0.18), (0.43, 1.48, 0.1),
        (1.63, 1.05, 1.84), (1.1, 0.53, 1.63), (0.7, 0.92, 0.62), (1.53, 1.84, 1.66), (1.65, 1.67, 2.91),
        (1.53, 0.22, 0.11), (1.94, 0.71, 0.08), (1.47, 1.41, 0.49), (1.47, 0.61, 1.62), (1.41, 0.22, 0.61),
        (1.12, 0.07, 2.4), (1.79, 1.06, 2.6), (0.95, 1.82, 2.06), (1.96, 0.84, 1.32), (0.67, 1.69, 2.75),
        (0.56, 0.25, 2.84), (1.95, 1.75, 0.04), (1.05, 1.64, 2.08), (0.88, 0.75, 2.32), (0.34, 0.9, 1.97),
        (1.3, 1.81, 0.98), (1.39, 0.94, 1.54), (1.98, 0.29, 2.99), (0.63, 1.56, 1.67), (1.39, 1.66, 2.96),
        (0.44, 0.99, 0.06), (1.17, 1.08, 2.12), (1.91, 0.07, 1.95), (0.56, 1.6, 1.24), (1.58, 0.35, 0.78),
        (1.56, 1.61, 1.25), (1.94, 1.58, 1.89), (1.38, 1.18, 2.62), (0.48, 1.61, 1.17), (1.94, 0.98, 1.96),
        (1.67, 0.54, 2.11), (0.07, 0.85, 0.26), (0.96, 1.07, 0.11), (0.71, 0.24, 0.52), (1.11, 0.51, 1.42),
        (1.2, 0.36, 1.19), (1.59, 0.31, 0.54), (1.06, 0.2, 2.84), (0.29, 1.22, 1.17), (0.0, 1.12, 2.74),
        (0.24, 1.09, 1.31), (0.14, 0.97, 2.37), (0.77, 1.64, 0.92), (0.76, 0.55, 1.75), (1.66, 1.78, 1.6),
        (1.9, 1.4, 2.97), (1.01, 0.03, 0.45), (0.97, 0.95, 2.39), (0.6, 1.27, 0.23), (0.16, 1.05, 1.61),
        (1.69, 0.14, 1.94), (1.46, 0.63, 2.25), (1.39, 1.61, 0.38), (1.31, 1.99, 2.12), (1.07, 0.73, 1.87),
        (1.18, 1.34, 2.08), (1.16, 1.38, 1.42), (0.97, 0.53, 1.91), (0.54, 0.59, 1.42), (1.51, 1.62, 2.91),
        (1.66, 0.51, 1.59), (0.5, 1.5, 0.14), (0.54, 0.93, 2.47), (0.65, 0.86, 1.68), (1.43, 0.15, 0.7)
    ])

    pkl_folder = 'data/fk11'
    radius, center = compute_tcp_positions(pkl_folder)

    if radius is not None and center is not None:
        chosen_centers, all_points = cover_points_with_ortools(apple_positions, radius, radius)

        if chosen_centers:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("X (m)")
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
                if any(np.linalg.norm(point - center) <= radius for center in chosen_centers) and point[2] < 2 * radius:
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

            ax.legend()
            plt.show()

            # 计算每个球体覆盖点集的最短路径
            for center in chosen_centers:
                covered_points_in_sphere = [p for p in covered_points if np.linalg.norm(p - center) <= radius]
                if covered_points_in_sphere:
                    solve_tsp(covered_points_in_sphere)
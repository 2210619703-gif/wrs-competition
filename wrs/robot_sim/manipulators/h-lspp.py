from lkh import LKH
import numpy as np

# 示例数据：生成10个城市的随机坐标
num_cities = 10
coordinates = np.random.rand(num_cities, 2) * 100  # 随机生成 0 到 100 范围的坐标

# 计算距离矩阵
dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        dist_matrix[i][j] = np.linalg.norm(coordinates[i] - coordinates[j])

# 初始化 LKH 求解器
solver = LKH(distance_matrix=dist_matrix)

# 运行求解器，获得最优路径和总距离
best_distance, best_route = solver.solve()

print("最佳路径长度：", best_distance)
print("最佳路径：", best_route)

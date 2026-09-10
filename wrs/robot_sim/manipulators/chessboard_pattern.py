import numpy as np
import cv2

# 定义棋盘格参数
CHESSBOARD_SIZE = (7, 9)  # 行列数
SQUARE_SIZE = 50  # 单个格子边长（单位：像素）

# 生成棋盘格
pattern_size = (CHESSBOARD_SIZE[0] + 1, CHESSBOARD_SIZE[1] + 1)
chessboard = np.zeros((pattern_size[1] * SQUARE_SIZE, pattern_size[0] * SQUARE_SIZE), dtype=np.uint8)

for i in range(pattern_size[1]):
    for j in range(pattern_size[0]):
        if (i + j) % 2 == 0:
            chessboard[i * SQUARE_SIZE:(i + 1) * SQUARE_SIZE, j * SQUARE_SIZE:(j + 1) * SQUARE_SIZE] = 255

# 保存棋盘格图片
cv2.imwrite("chessboard.png", chessboard)

# 显示棋盘格
cv2.imshow("Chessboard", chessboard)
cv2.waitKey(0)
cv2.destroyAllWindows()

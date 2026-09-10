import pickle

def load_pointcloud_data(filename="pointcloud_20250529_214124.pkl"):
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"[INFO] Successfully loaded data from '{filename}'")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load data from '{filename}': {e}")
        return None


# 使用函数读取文件
pointcloud_data = load_pointcloud_data("pointcloud_20250922_161611.pkl")

# 打印加载的数据
if pointcloud_data:
    print(f"[INFO] Loaded {len(pointcloud_data)} frames.")
    # 如果数据较大，可以查看前几项内容
    print(pointcloud_data)
else:
    print("[INFO] No data found.")

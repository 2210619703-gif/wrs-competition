import os
import pickle

def load_pkl_files(directory):
    data_list = []  # 创建一个空列表来存储所有文件的数据
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data_list.append(data)
    return data_list

if __name__ == "__main__":
    # 设置 pkl 文件所在目录
    directory = 'data/fk11'  # 替换为您的目录路径
    data_list = load_pkl_files(directory)
    # 现在 data_list 包含了所有 pkl 文件的数据
    for i, data in enumerate(data_list):
        print(f"Data from file {i+1}: {data}")

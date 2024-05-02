import os
import numpy as np
import pandas as pd


def split(seeg_index_path, sub_paths, save_path):
    # 确保新文件夹存在
    os.makedirs(save_path, exist_ok=True)

    for seeg_path in sub_paths:
        for npy_filename in os.listdir(seeg_path):
            # 获取电影编号
            movie_num = int(npy_filename.split('_')[1])  # 假设文件名格式为movie_i_first.npy
            # 获取观看次序
            watch_order = npy_filename.split('_')[2].split('.')[0]  # 假设文件名格式为movie_i_first.npy

            # 构建对应的csv文件路径
            index_file = f"{seeg_index_path}/movie_{movie_num}_index.csv"

            # 检查csv文件是否存在
            if os.path.exists(index_file):
                # 加载.npy文件
                npy_data = np.load(os.path.join(seeg_path, npy_filename)).T

                # 读取CSV文件
                csv_data = pd.read_csv(index_file)

                # 创建新的文件夹来存储结果
                result_folder = os.path.join(save_path, f"movie_{movie_num}")
                os.makedirs(result_folder, exist_ok=True)

                # 遍历CSV文件中的每一行
                for _, row in csv_data.iterrows():
                    # 获取clip_i值
                    clip_i = row['clip_i']
                    # 获取起始和结束索引
                    start_index = row['index_start']
                    end_index = row['index_end']

                    # 根据索引提取对应的列
                    extracted_array = npy_data[:, start_index:end_index]

                    # 构建新的.npy文件名
                    new_npy_file_name = f"movie_{movie_num}_{watch_order}_{clip_i}.npy"
                    # 保存提取的数组为.npy文件
                    np.save(os.path.join(result_folder, new_npy_file_name), extracted_array)
            else:
                print(f"CSV文件 {index_file} 不存在。")


if __name__ == "__main__":
    # 设置路径
    sub_paths = ['./seeg_raw/sub_07/sub_07_npy']
    seeg_index_path = './seeg_index'
    save_path = 'seeg_split/sub_07'
    split(seeg_index_path, sub_paths, save_path)

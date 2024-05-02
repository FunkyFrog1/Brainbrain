import os
import numpy as np
import pandas as pd


def split(movie_eeg_index_path, sub_paths):
    for seeg_path in sub_paths:
        for npy_filename in os.listdir(seeg_path):
            #获取被试编号，创建保存结果的路径
            sub_num = os.path.basename(seeg_path)
            # 查找最后一个下划线的位置
            last_underscore_index = sub_num.rfind('_')
            # 提取第二个下划线之前的字符
            sub_num = sub_num[:last_underscore_index]
            result_path = f'F:/Brain/data/{sub_num}/{sub_num}_split'
            if npy_filename.endswith('.npy') and npy_filename.startswith('movie'):
                # 获取i值
                i = int(npy_filename.split('_')[1])  # 假设文件名格式为movie_i_first.npy
                watch_order = npy_filename.split('_')[2].split('.')[0]  # 假设文件名格式为movie_i_first.npy

                # 构建对应的csv文件路径
                csv_filename = f"movie_{i}_index.csv"
                csv_filepath = os.path.join(movie_eeg_index_path, csv_filename)

                # 检查csv文件是否存在
                if os.path.exists(csv_filepath):
                    # 加载.npy文件
                    npy_data = np.load(os.path.join(seeg_path, npy_filename)).T

                    # 读取CSV文件
                    csv_data = pd.read_csv(csv_filepath)

                    # 创建新的文件夹来存储结果
                    result_folder = os.path.join(result_path, f"movie_{i}")
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
                        new_npy_file_name = f"movie_{i}_{watch_order}_{clip_i}.npy"
                        # 保存提取的数组为.npy文件
                        np.save(os.path.join(result_folder, new_npy_file_name), extracted_array)
                else:
                    print(f"CSV文件 {csv_filename} 不存在。")


# 设置路径
sub_paths = ['F:/Brain/data/sub_01/sub_01_npy', 'F:/Brain/data/sub_07/sub_07_npy',
             'F:/Brain/data/sub_10/sub_10_npy', 'F:/Brain/data/sub_12/sub_12_npy']
movie_eeg_index_path = './movie_eeg_index'
split(movie_eeg_index_path, sub_paths)

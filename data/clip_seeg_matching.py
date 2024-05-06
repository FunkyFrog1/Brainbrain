import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import h5py


# 归一化数据
def normalize(x):
    min_val = np.amin(x)
    max_val = np.amax(x)
    x = (x - min_val) / (max_val - min_val)
    return x


# 根据下划线获得文件名中的内容
def extract_content_by_underscores(file_name):
    # 查找第一个下划线的位置
    first_underscore_index = file_name.find("_")

    # 查找第二个下划线的位置
    second_underscore_index = file_name.find("_", first_underscore_index + 1)

    # 查找第三个下划线的位置
    third_underscore_index = file_name.find("_", second_underscore_index + 1)

    # 查找第四个下划线的位置
    fourth_underscore_index = file_name.find("_", third_underscore_index + 1)

    # 获取电影编号
    movie_num = file_name[first_underscore_index + 1:second_underscore_index]

    # 获取clip编号
    if fourth_underscore_index == -1:
        clip_num = int(file_name[third_underscore_index + 1:file_name.rfind(".")])
        clip_num_sub = -1
    else:
        clip_num = int(file_name[third_underscore_index + 1:fourth_underscore_index])
        clip_num_sub = int(file_name[fourth_underscore_index + 1:file_name.rfind(".")])

        # 获取观看顺序标签 first：0 second: 1
    watch_order = file_name[second_underscore_index + 1:third_underscore_index]

    if watch_order == 'first':
        watch_flag = 0
    else:
        watch_flag = 1

    return movie_num, clip_num, clip_num_sub, watch_flag


# 根据索引获取训练集和测试集
def split_data(data_list, train_indices, test_indices):
    train_data = [[data[i] for i in train_indices] for data in data_list]
    test_data = [[data[i] for i in test_indices] for data in data_list]
    return train_data, test_data


# 遍历所有seeg数据 与first_frame进行对齐
def clip_seeg_matching(movie_nums, sub_dir, sub_num):
    data_list = [[],[],[],[],[],[]]
    bad_data = 0
    for movie_num in movie_nums:
        for file in os.listdir(os.path.join(sub_dir, f'movie_{movie_num}')):
            # 加载seeg数据
            seeg_matrix = np.load(os.path.join(sub_dir, f'movie_{movie_num}', file))

            if seeg_matrix.shape[1] != 500:
                print(seeg_matrix.shape)
                bad_data += 1
                continue

            # 归一化seeg数据
            seeg_matrix = np.array(normalize(seeg_matrix).astype(np.float32))

            _, clip_num, clip_num_sub, watch_flag = extract_content_by_underscores(file)
            # 加载并归一化pfm矩阵, 并生成向量
            # 打开图片文件
            if clip_num_sub == -1:
                img = Image.open(f'./clip_croped_first_frame/movie_{movie_num}/clip_{clip_num}.jpg')
            else:
                img = Image.open(f'./clip_croped_first_frame/movie_{movie_num}/clip_{clip_num}_{clip_num_sub}.jpg')
            # 将图片转换为numpy数组,形状为（3，512，512）
            first_frame_matrix = np.array(img).transpose((2, 0, 1))
            # 将seeg_matrix、first_frame_matrix、movie_num，clip_num,clip_num_sub,watch_flag存储到列表中
            data_list[0].append(seeg_matrix)
            data_list[1].append(first_frame_matrix)
            data_list[2].append(movie_num)
            data_list[3].append(clip_num)
            data_list[4].append(clip_num_sub)
            data_list[5].append(watch_flag)

    print(bad_data)

    # 总样本个数
    n_samples = len(data_list[0])

    # 生成样本的索引
    indices = np.arange(n_samples)

    # 划分训练集和测试集的索引
    # train_indices, test_indices = train_test_split(indices, train_size=0.9, test_size=0.1, random_state=42)

    # train_data_list, test_data_list = split_data(data_list, train_indices, test_indices)

    # print(type(test_data_list[0]))
    # print(type(test_data_list[1][0]))
    # print(type(test_data_list[2][0]))
    # print(type(test_data_list[3][0]))
    # print(type(test_data_list[4][0]))
    # print(type(test_data_list[5][0]))

    # 将训练集存储到一个新的hdf5文件
    with h5py.File(f'sub_{sub_num}_data.h5', 'w') as f:
        for i, name in enumerate(['seeg', 'first_frame', 'movie_num', 'clip', 'clip_sub', 'watch_flag']):
            f.create_dataset(name, data=data_list[i])

    # # 将测试集存储到另一个新的hdf5文件
    # with h5py.File(f'sub_{sub_num}_test_data.h5', 'w') as f:
    #     for i, name in enumerate(['seeg', 'first_frame', 'movie_num', 'clip', 'clip_sub', 'watch_flag']):
    #         f.create_dataset(name, data=test_data_list[i])


if __name__ == '__main__':
    movie_nums_01 = [9,10,11,17,18,23,29,34]
    movie_nums_07 = [0,7,11,19,25,31,32,35]
    movie_nums_10 = [3,7,8,14,21,26,29,32]
    movie_nums_12 = [0,7,9,11,17,29,33,34]
    sub_dir_01 = './sub_01/sub_01_split'
    sub_dir_07 = 'seeg_split/sub_07'
    sub_dir_10 = './sub_10/sub_10_split'
    sub_dir_12 = './sub_12/sub_12_split'
    sub_num_01 = '01'
    sub_num_07 = '07'
    sub_num_10 = '10'
    sub_num_12 = '12'
    clip_seeg_matching(movie_nums_07, sub_dir_07, sub_num_07)


import pandas as pd
import os

# 指定包含原始CSV文件的文件夹路径
original_folder_path = 'movie_info'
# 指定保存修改后CSV文件的新文件夹路径
new_folder_path = 'movie_eeg_index'

# 确保新文件夹存在
os.makedirs(new_folder_path, exist_ok=True)

# 遍历原始文件夹中的所有文件
for file_name in os.listdir(original_folder_path):
    # 检查文件是否符合原始CSV文件的命名模式
    if file_name.startswith('info_') and file_name.endswith('.csv'):
        # 构建原始CSV文件的完整路径
        original_file_path = os.path.join(original_folder_path, file_name)
        # 读取CSV文件
        df = pd.read_csv(original_file_path)

        # 执行计算并保留整数部分
        df['index_start'] = (df['start_time'] * 500).astype(int)
        df['index_end'] = (df['end_time'] * 500).astype(int)

        # 删除原始的 'start_time' 和 'end_time' 列
        df.drop(['start_time', 'end_time'], axis=1, inplace=True)

        # 重命名列
        df.rename(columns={'clip_i': 'clip_i'}, inplace=True)

        # 构建新CSV文件的名称和完整路径
        new_file_name = file_name.replace('info_', 'movie_').replace('.csv', '_index.csv')
        new_file_path = os.path.join(new_folder_path, new_file_name)
        # 保存修改后的CSV文件到新文件夹
        df.to_csv(new_file_path, index=False)



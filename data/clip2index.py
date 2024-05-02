import pandas as pd
import os


def get_seeg_index(time_info_path, save_path):
    # 确保新文件夹存在
    os.makedirs(save_path, exist_ok=True)
    # 遍历原始文件夹中的所有文件
    for file_name in os.listdir(time_info_path):
        # 构建原始CSV文件的完整路径
        file_path = os.path.join(time_info_path, file_name)
        # 读取CSV文件
        df = pd.read_csv(file_path)

        # 执行计算并保留整数部分
        df['index_start'] = (df['start_time'] * 500).astype(int)
        df['index_end'] = (df['end_time'] * 500).astype(int)

        # 删除原始的 'start_time' 和 'end_time' 列
        df.drop(['start_time', 'end_time'], axis=1, inplace=True)

        # 构建新CSV文件的名称和完整路径
        new_file_name = file_name.replace('_info', '_index')
        new_file_path = os.path.join(save_path, new_file_name)
        # 保存修改后的CSV文件到新文件夹
        df.to_csv(new_file_path, index=False)


if __name__ == "__main__":
    # 指定包含原始CSV文件的文件夹路径
    info_path = './clip_croped/info'
    # 指定保存修改后CSV文件的新文件夹路径
    save_path = 'seeg_index'
    get_seeg_index(info_path, save_path)
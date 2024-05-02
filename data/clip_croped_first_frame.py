import os
import shutil
import cv2
from tqdm import tqdm


# 保存视频第一帧
def save_first_frame(clip_path, clip_save_path):
    # 修改后缀为.jpg格式
    clip_save_path, _ = os.path.splitext(clip_save_path)
    clip_save_path = f"{clip_save_path}.jpg"

    # 打开视频文件
    video = cv2.VideoCapture(clip_path)

    # 读取第一帧
    _, frame = video.read()

    cv2.imwrite(clip_save_path, frame)

    # 释放资源
    video.release()

    return frame


def main():
    # 输入视频文件夹路径和输出文件夹路径
    src = 'clip_croped'  # 视频文件夹路径
    dst = 'clip_croped_first_frame'  # 输出文件夹路径
    movie_num = 36  # 视频数量

    # 如果输出文件夹不存在，创建它
    if not os.path.exists(dst):
        os.makedirs(dst)

    # 如果info文件夹存在，复制一份到dst目录
    info_folder_path = os.path.join(src, 'info')
    if os.path.exists(info_folder_path) and os.path.isdir(info_folder_path):
        # 复制 'info' 文件夹到目标文件夹
        shutil.copytree(info_folder_path, os.path.join(dst, 'info'))

    # 遍历每个视频
    for i in tqdm(range(movie_num)):
        movie_path = os.path.join(src, f'movie_{i}')
        movie_save_path = os.path.join(dst, f'movie_{i}')

        for clip in os.listdir(movie_path):
            clip_path = os.path.join(movie_path, clip)
            clip_save_path = os.path.join(movie_save_path, clip)

            # 如果保存目录不存在，创建它
            if not os.path.exists(movie_save_path):
                os.makedirs(movie_save_path)

            save_first_frame(clip_path, clip_save_path)  # 保存视频第一帧


if __name__ == '__main__':
    main()

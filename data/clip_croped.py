import os
import cv2
import shutil
import pandas as pd
import subprocess

from skimage import transform, util
import numpy as np


# 设置裁切时长
DURATION = 1
INTERVAL = 1


# 裁剪并降采样帧
def crop_downsample(img, size=(512, 512)):
    y, x = img.shape[:2]
    startx = x // 2 - min(x, y) // 2
    starty = y // 2 - min(x, y) // 2
    # 获取裁切后的图片
    img = img[starty:starty + min(x, y), startx:startx + min(x, y)]
    # 获取降采样后的图片
    img = transform.resize(img, size)
    return img


# 获取视频裁剪尺寸
def get_video_crop(file_path):
    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    crop = [height, (width-height)/2]
    return crop


# 裁剪视频为512px
def crop_video(input_path, output_path):
    # 如果路径不存在，创建路径
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    crop = get_video_crop(input_path)
    command = f"ffmpeg -i {input_path} -t {DURATION} -filter:v \"crop={crop[0]}:{crop[0]}:{crop[1]}:0,scale=512:512\" {output_path}"
    subprocess.call(command, shell=True)


def sequential_cutting(input_path, output_path, clip_i, info, interval=INTERVAL):
    # 获取开始结束时间
    start_time = info.loc[clip_i, 'start_time']
    end_time = info.loc[clip_i, 'end_time']
    # 如果路径不存在，创建路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cutting_list = []
    crop = get_video_crop(input_path)
    # 时间指针
    pointer = start_time
    while pointer + DURATION <= end_time:
        cutting_list.append(pointer - start_time)
        pointer += interval
    
    # 用attach_info来保存新的信息
    attach_info = pd.DataFrame({})
    # 序列式裁切视频
    for i, p in enumerate(cutting_list):
        full_output_path = os.path.join(output_path, f'clip_{clip_i}_{i}.mp4')
        command = f"ffmpeg -i {input_path} -ss {p} -t {DURATION} -filter:v \"crop={crop[0]}:{crop[0]}:{crop[1]}:0,scale=512:512\" {full_output_path}"
        subprocess.call(command, shell=True)
        # 处理info.csv
        df = info.loc[clip_i].copy()
        df.loc["start_time"] = df.loc['start_time'] + p
        df.loc['end_time'] = df.loc["start_time"] + DURATION
        df.loc['clip_i'] = df.loc['clip_i'] + f'_{i}'
        attach_info = attach_info._append(df, ignore_index=True)

    # 返回需要添加的info
    #print(attach_info)
    return attach_info
        

# 视频预处理
def video_preprocess(src, dst):
    for movie in os.listdir(src):

        # 读取info文件
        movie_path = os.path.join(src, movie)
        info = pd.read_csv(os.path.join(movie_path, 'info.csv'), dtype={'clip_i':str})

        # 用这个来存储新的info文件，过滤没有用的info
        new_info = pd.DataFrame({})

        for clips in os.listdir(movie_path):
            # 如果是2-3s，就裁切前2s
            if clips == '1_2s':
                clips_path = os.path.join(movie_path, clips)
                for clip in os.listdir(clips_path):
                    # 获取输入路径
                    input_path = os.path.join(clips_path, clip)
                    # 获取输出路径
                    output_path = os.path.join(dst, os.path.relpath(input_path, src))
                    # 裁剪视频
                    crop_video(input_path, output_path)
                    # 获取视频编号，并修改info，将修改后的info添加到new_info中
                    clip_i = int((clip[5:])[:-4])
                    info.loc[clip_i, 'end_time'] = info.loc[clip_i, 'start_time'] + 2
                    new_info = new_info._append(info.loc[clip_i])

            # 如果视频大于3s，另作处理
            elif (clips == '2_3s') or (clips == '3+s'):
                clips_path = os.path.join(movie_path, clips)
                for clip in os.listdir(clips_path):
                    # 获取输入路径
                    input_path = os.path.join(clips_path, clip)
                    # 获取输出路径
                    output_path = os.path.join(dst, os.path.relpath(clips_path, src))
                    # 获取视频编号
                    clip_i = int((clip[5:])[:-4])
                    attach_info = sequential_cutting(input_path, output_path, clip_i, info)
                    new_info = pd.concat([new_info, attach_info], ignore_index=True)

        # 保存新的info.csv        
        new_info.to_csv(os.path.join(dst, movie, 'info.csv'), index=False)


def main():
    src = 'clip_raw'  # 源文件夹路径
    dst = 'clip_croped'  # 目标文件夹路径

    video_preprocess(src, dst)


if __name__ == '__main__':
    main()

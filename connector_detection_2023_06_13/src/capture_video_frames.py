#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""本模块用于对视频文件夹中所有视频，提取其中的画面帧，并把结果保存在视频所在的文件夹中。

版本号： 0.2
日期： 2023-06-25
作者： jun.liu@leapting.com
"""
import os
import pathlib
import shutil
import sys

import cv2 as cv
from tqdm import tqdm


def capture_one_video(video_path, capture_frames_per_second, frames_folder):
    """capture one frame.

    Arguments：
        video_path (pathlib.Path): 一个 Path 对象，是视频的路径。
        capture_frames_per_second (int | float): 一个整数或小数，是每秒需要提取的图像帧数。
        frames_folder (int): 一个整数，是每秒需要提取的图像帧数。
    """
    if not video_path.exists():
        raise FileNotFoundError(f'{video_path} does not exist!')

    capture = cv.VideoCapture(str(video_path))  # noqa

    fps = capture.get(cv.CAP_PROP_FPS)  # noqa  帧率
    # 每经过 capture_every_frames 帧图像，则提取一次。
    capture_every_frames = int(fps / capture_frames_per_second)
    if not capture.isOpened():
        print(f'Unable to open {video_path}')
        sys.exit()

    captured_frames_tally = 0
    total_frames = capture.get(cv.CAP_PROP_FRAME_COUNT)  # noqa 总的帧数 CAP_PROP_FRAME_COUNT
    with tqdm(total=total_frames, desc='processing frames', unit=' frames',
              leave=True, ncols=100) as progress_bar:
        while True:
            progress_bar.update(1)  # 更新进度条。
            _, frame = capture.read()
            if frame is None:
                break  # 视频结束时， frame 为 None 值。
            else:
                frames_quantity = capture.get(cv.CAP_PROP_POS_FRAMES)  # noqa 当前帧的计数值

                frame_name = video_path.stem + f'_{captured_frames_tally:03}.jpg'
                frames_path = frames_folder / frame_name
                if frames_quantity % capture_every_frames == 0:
                    captured_frames_tally += 1
                    cv.imwrite(str(frames_path), frame)
    capture.release()
    print(f'{captured_frames_tally} frames captured, \nsaved to {frames_folder}\n')

    return captured_frames_tally


def main(video_path, capture_frames_per_second):
    """对视频文件夹 :attr:`video_path` 中所有视频，提取其中的画面帧，结果保存在视频文件夹中。

    Arguments：
        video_path (pathlib.Path): 一个 Path 对象，指向一个存放视频的文件夹，其中可以存放多个视频。
        capture_frames_per_second (int | float): 一个整数或小数，是每秒需要提取的图像帧数。
    """
    video_path = pathlib.Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f'{video_path} does not exist!')

    frames_tally = 0
    # tqdm_video = tqdm(video_path.iterdir(), total=len(os.listdir(video_path)), ncols=80)
    for each in video_path.iterdir():
        if each.is_file():
            # 对每个视频文件创建一个文件夹。
            frames_folder = video_path / f'captured_frames_{each.stem}'
            if frames_folder.exists():  # 如果之前存在该文件夹，则先删除
                shutil.rmtree(frames_folder)
            frames_folder.mkdir()
            print(f'Processing {each.name} :')
            frames_in_one_video = capture_one_video(
                video_path=each,
                capture_frames_per_second=capture_frames_per_second,
                frames_folder=frames_folder)
            frames_tally += frames_in_one_video

    print(f'Done! Total {frames_tally} frames captured.')


if __name__ == '__main__':
    video_path = r'/media/drin/shared_disk/work/cv/2023_06_07_connector/videos'
    main(video_path=video_path, capture_frames_per_second=2)

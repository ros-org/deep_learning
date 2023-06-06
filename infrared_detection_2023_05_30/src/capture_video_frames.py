#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""本模块用于对视频文件夹中所有视频，提取其中的画面帧，并把结果保存在视频文件夹中。

版本号： 0.1
日期： 2023-06-05
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

    frames_tally = 0
    while True:
        _, frame = capture.read()
        if frame is None:
            break  # 视频结束时， frame 为 None 值。
        else:
            frames_quantity = capture.get(cv.CAP_PROP_POS_FRAMES)  # noqa 当前帧的计数值

            frame_name = video_path.stem + f'_{frames_tally:03}.jpg'
            frames_path = frames_folder / frame_name
            if frames_quantity % capture_every_frames == 0:
                frames_tally += 1
                cv.imwrite(str(frames_path), frame)

    capture.release()
    return frames_tally


def main(video_path, capture_frames_per_second):
    """对视频文件夹 :attr:`video_path` 中所有视频，提取其中的画面帧，结果保存在视频文件夹中。

    Arguments：
        video_path (pathlib.Path): 一个 Path 对象，指向一个存放视频的文件夹，其中可以存放多个视频。
        capture_frames_per_second (int | float): 一个整数或小数，是每秒需要提取的图像帧数。
    """
    video_path = pathlib.Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f'{video_path} does not exist!')

    frames_folder = video_path / 'captured_frames'
    if frames_folder.exists():  # 如果之前存在该文件夹，则先删除
        shutil.rmtree(frames_folder)
    frames_folder.mkdir()
    frames_tally = 0
    tqdm_video = tqdm(video_path.iterdir(), total=len(os.listdir(video_path)), ncols=80)
    for each in tqdm_video:
        if each.is_file():
            frames_in_one_video = capture_one_video(
                video_path=each,
                capture_frames_per_second=capture_frames_per_second,
                frames_folder=frames_folder)
            frames_tally += frames_in_one_video

    print(f'Done! Total {frames_tally} frames captured, \nsaved to {frames_folder}')


if __name__ == '__main__':
    video_path = r'~/work/cv/2023_05_24_infrared/dataset_solar_panel/2023-06-05 湖州红外视频/采用的视频'
    main(video_path=video_path, capture_frames_per_second=0.5)

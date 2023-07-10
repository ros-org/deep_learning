#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""把图片重新保存一遍，因为运行 YOLOv8 时报错： corrupt JPEG restored and saved。"""

import concurrent.futures
import os
import pathlib

import cv2
from tqdm import tqdm

import utilities


def _open_save_img(input_path: pathlib.Path, original_images_path: pathlib.Path):
    if input_path.is_file():
        img = cv2.imread(str(input_path))  # noqa
        copy_to = original_images_path / input_path.name  # copy_to 是新图片的存放位置
        cv2.imwrite(str(copy_to), img)  # noqa


@utilities.timer
def fix_images(original_images_path):

    original_images_path = pathlib.Path(original_images_path).expanduser().resolve()
    if original_images_path.exists():
        print(f'{original_images_path= }')
    else:
        raise FileNotFoundError(f'{original_images_path= }')

    new_name = original_images_path.name + '_obsolete'
    obsolete_folder = original_images_path.parent / new_name
    original_images_path.rename(obsolete_folder)  # 给文件夹改名字

    original_images_path.mkdir()  # 原始文件夹被改名后，需要重新创建一次。

    original_images_path_generator = [original_images_path] * len(os.listdir(obsolete_folder))
    # 使用 ThreadPoolExecutor 进行并发，速度提高约 9 倍，所用时间 从 65s 降为 7s （320 张图片）
    print('Fixing images ...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:  # noqa
        results = executor.map(_open_save_img,
                               obsolete_folder.iterdir(), original_images_path_generator)

        # 为了显示一个 tqdm 进度条，使用下面这个循环。注意必须在 ThreadPoolExecutor 之内进行循环。
        tqdm_results = tqdm(results, total=len(os.listdir(obsolete_folder)), ncols=80)
        for result in tqdm_results:
            pass  # 显示进度条，无须其它操作，直接 pass 即可。

    print(f'Fixed images: {len(os.listdir(obsolete_folder))}')


if __name__ == '__main__':
    train_path = "/media/disk_2/work/cv/2023_06_07_connector/dataset_connector/images/train"
    val_path = "/media/disk_2/work/cv/2023_06_07_connector/dataset_connector/images/validation"
    test_path = "/media/disk_2/work/cv/2023_06_07_connector/dataset_connector/images/test"
    images_paths = train_path, val_path, test_path

    for image_path in images_paths:
        print(f'{image_path= }')
        fix_images(original_images_path=image_path)



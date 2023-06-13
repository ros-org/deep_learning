#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""把图片重新保存一遍，因为运行 YOLOv8 时报错： corrupt JPEG restored and saved。"""

import os
import pathlib

import cv2
from tqdm import tqdm


def fix_images(original_iamges_path, fixed_images_path):

    sorted_images = sorted(os.listdir(original_iamges_path))
    tqdm_images = tqdm(sorted_images, total=len(os.listdir(original_iamges_path)), ncols=80)

    for k, pic in enumerate(tqdm_images, 0):
        filep = os.path.join(original_iamges_path, pic)
        img = cv2.imread(filep)  # noqa
        cv2.imwrite(os.path.join(fixed_images_path, pic), img)

    print(f'Fixed images {len(os.listdir(fixed_images_path))}')


if __name__ == '__main__':
    train_path = "/media/drin/shared_disk/work/cv/2023_06_07_connector/dataset_connector/images/train"
    fixed_train_path = "/media/drin/shared_disk/work/cv/2023_06_07_connector/dataset_connector/images/fixed/train"

    val_path = "/media/drin/shared_disk/work/cv/2023_06_07_connector/dataset_connector/images/validation"
    fixed_val_path = "/media/drin/shared_disk/work/cv/2023_06_07_connector/dataset_connector/images/fixed/validation"

    test_path = "/media/drin/shared_disk/work/cv/2023_06_07_connector/dataset_connector/images/test"
    fixed_test_path = "/media/drin/shared_disk/work/cv/2023_06_07_connector/dataset_connector/images/fixed/test"

    images_paths = train_path, val_path, test_path
    fixed_paths = fixed_train_path, fixed_val_path, fixed_test_path

    for each_path in fixed_paths:
        each_path = pathlib.Path(each_path).expanduser().resolve()
        if not each_path.exists():
            each_path.mkdir(parents=True)

    for image_path, fixed_path in zip(images_paths, fixed_paths):
        print(f'{image_path= }')
        fix_images(original_iamges_path=image_path, fixed_images_path=fixed_path)


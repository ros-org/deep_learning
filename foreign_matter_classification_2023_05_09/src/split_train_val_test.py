#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于把输入文件夹内的图片，按比例复制到 3 个文件夹： train, validation, test。

使用方法：终端运行 split_dataset(image_path, class_type='foreign_matter')

默认的 train, validation, test 分配比例是 3:1:1 。可以手动修改这个比例值。

版本号： 0.1
日期： 2023-05-11
作者： jun.liu@leapting.com
"""
import pathlib
import shutil

import numpy as np


def copy_one_image(each, processed, copied, tally,
                   train_quantity, validation_quantity, test_quantity,
                   train_dir_classified, validation_dir_classified, test_dir_classified):
    processed += 1  # 计算处理过的数量。
    if processed <= train_quantity:
        shutil.copy(each, train_dir_classified)  # 尝试改为 copy
        copied += 1
    elif processed == train_quantity + validation_quantity:
        shutil.copy(each, validation_dir_classified)
        copied += 1
    elif processed == train_quantity + validation_quantity + test_quantity:
        shutil.copy(each, test_dir_classified)
        copied += 1
        processed = 0  # 注意必须在 processed = 5 时归零，重新累计。
    tally += 1  # 计算处理过的数量。
    return processed, copied, tally


def split_dataset(image_path, class_type, ratio=(3, 1, 1)):
    """把输入文件夹 :attr:`image_path` 内的图片，按比例复制到 3 个文件夹： train, validation, test。

    Arguments:
        image_path (str): 一个字符串，指向一个文件夹，其中放着图片文件。字符串中必须包含 foreign_matter
            或 OK 文件夹，即类别文件夹的名字。后面会检查 class_type 是否和 image_path 的类别文件夹一致。
            在文件夹中，允许包含子文件夹，会扫描子文件夹内的图片，并且会把子文件夹的图片也并分配到 3 个数据集
            中去。但是不会处理“孙文件夹”（即如果子文件夹 foo 中再包含下一级子文件夹 bar，则子文件夹 bar 会
            被忽略，不再扫描 bar 中的图片）。
        class_type (str): 一个字符串，是 'OK' 或 'foreign_matter'，表示分类的类别。
        ratio (tuple[int, int, int]): 一个元祖，其中包含 3 个整数，分别表示训练集，验证集和测试集的划分比例。
    """
    if class_type not in ['OK', 'foreign_matter']:
        raise ValueError('The class_type must be one of "OK, foreign_matter".')
    elif class_type not in image_path:
        raise ValueError(f"The {class_type=}, which doesn't match the image_path: {image_path} !")

    # 检查是否每个数值的都是整数
    int_check = np.asarray([isinstance(each, int) for each in ratio])
    if len(ratio) != 3:
        raise ValueError('The ratios must be a tuple, '
                         'format: (train_ratio, validation_ratio, train_ratio)!')
    elif not np.all(np.asarray(ratio) >= 1):  # todo: 增加 ratio 为整数的检查。
        raise ValueError(f'The ratio must in range [0, 1] ! Current value: {ratio=}')
    elif not np.all(int_check):  # 检查数值类型
        raise ValueError(f'The ratio value must be an integer ! Current value: {ratio=}')

    image_path = pathlib.Path(image_path).expanduser().resolve()

    # dataset_root 用于存放训练、测试等 3 个数据集。
    dataset_root = pathlib.Path(
        r'~/work/cv/2023_05_04_regnet/classification_data').expanduser().resolve()

    train_dir_classified = dataset_root / 'train' / class_type  # 加上类别。
    validation_dir_classified = dataset_root / 'validation' / class_type
    test_dir_classified = dataset_root / 'test' / class_type
    # 如果文件夹不存在，则创建该文件夹。
    if not train_dir_classified.exists():
        train_dir_classified.mkdir(parents=True)
    if not validation_dir_classified.exists():
        validation_dir_classified.mkdir(parents=True)
    if not test_dir_classified.exists():
        test_dir_classified.mkdir(parents=True)

    train_quantity = ratio[0]
    validation_quantity = ratio[1]
    test_quantity = ratio[2]
    print(f'Dataset split ratio: {train_quantity=}, '
          f'{validation_quantity=}, {test_quantity=}')

    processed = 0
    tally = 0
    copied = 0
    for each in sorted(image_path.iterdir()):  # iterdir 默认是乱序的，所以要用 sorted 。
        if each.is_dir():
            for one_image in each.iterdir():
                processed, copied, tally = copy_one_image(
                    one_image, processed, copied, tally,
                    train_quantity, validation_quantity, test_quantity,
                    train_dir_classified, validation_dir_classified, test_dir_classified)
        elif each.is_file():  # 忽略文件夹。
            processed, copied, tally = copy_one_image(
                each, processed, copied, tally,
                train_quantity, validation_quantity, test_quantity,
                train_dir_classified, validation_dir_classified, test_dir_classified)

    missing = tally - copied
    assert (missing == 0), f'Alert! Some images are not moved, quantity: {missing}'
    print(f'Images quantity: {tally}')
    print('Done!')


if __name__ == '__main__':
    image_path = r'~/work/cv/2023_05_04_regnet/classification_data/original_data/foreign_matter/'
    # image_path = r'~/work/cv/2023_05_04_regnet/classification_data/original_data/OK/'
    split_dataset(image_path, class_type='foreign_matter')


#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于把输入文件夹内的图片，按比例复制到 3 个文件夹： train, validation, test。主要适用于
 YOLOv8 的物体检测 object detection 任务。

在分配到 3 个数据集之前，会先根据 json 标注文件，生成 txt 标签文件。
默认的 train, validation, test 分配比例是 3:1:1 。可以手动修改这个比例值。

版本号： 0.4
日期： 2023-06-13
作者： jun.liu@leapting.com
"""
import os
import pathlib
import shutil

import numpy as np
from tqdm import tqdm

from json2txt_yolov8 import get_polygons_for_all_jsons


def copy_one_file(each, processed, copied, tally,
                  train_quantity, validation_quantity, test_quantity,
                  train_dir, validation_dir, test_dir):
    """拷贝单个文件，可以是图片或者标签。按照 :attr:`processed` 计数值，分配到 3 个不同的文件夹中。"""
    processed += 1  # 计算处理过的数量。
    if processed <= train_quantity:
        # 为了保留 metadata，使用 copy2()。 YOLOv8 在进行预测时，
        # 会发生报错： corrupt JPEG restored and saved . 但是使用 copy2() 似乎并没有什么帮助？
        shutil.copy2(each, train_dir)
        copied += 1
    elif processed <= train_quantity + validation_quantity:
        shutil.copy2(each, validation_dir)
        copied += 1
    elif processed <= train_quantity + validation_quantity + test_quantity:
        shutil.copy2(each, test_dir)
        copied += 1
        if processed == train_quantity + validation_quantity + test_quantity:
            processed = 0  # 注意必须在 processed = 5 时归零，重新累计。
    tally += 1  # 计算处理过的数量。
    return processed, copied, tally


def extract_txt_and_split_dataset(images_jsons_folder, images_jsons_in_one=False,
                                  images_labels_folder=None,
                                  ratio=(3, 1, 1),
                                  overwrite_annotations=True, overwrite_images_labels=False):
    """把 json 标注文件转换为 txt 标签文件，然后把输入文件夹 :attr:`images_jsons_folder` 内的图片和
    txt 标签文件，按比例复制到 3 个文件夹：train, validation, test。

    Arguments:
        images_jsons_folder (str): 一个字符串，指向一个文件夹，其中放着 2 个文件夹，分别是 images 和 jsons。
        images_jsons_in_one (bool): 一个?。
        images_labels_folder (str): 一个字符串，指向一个文件夹，其中???。
        ratio (tuple[int, int, int]): 一个元祖，其中包含 3 个整数，分别表示训练集，验证集和测试集的划分比例。
        overwrite_annotations (bool): 一个布尔值，如果为 True，则会把之前的 txt 标注文件删除，然后重新生成标注文件。
        overwrite_images_labels (bool): 一个布尔值，如果为 True，则会把之前的图片和标签删除，然后重新生成。
    """
    images_jsons_folder = pathlib.Path(images_jsons_folder).expanduser().resolve()
    if images_jsons_in_one:
        raw_images = images_jsons_folder
        raw_jsons = images_jsons_folder
    else:
        raw_images = images_jsons_folder / 'images'
        raw_jsons = images_jsons_folder / 'jsons'
    if not raw_images.exists():
        raise FileNotFoundError(f'The folder does not exist: {raw_images}')
    if not raw_jsons.exists():
        raise FileNotFoundError(f'The folder does not exist: {raw_jsons}')

    # 检查是否每个数值的都是整数
    int_check = np.asarray([isinstance(each, int) for each in ratio])
    if len(ratio) != 3:
        raise ValueError('The ratios must be a tuple, '
                         'format: (train_ratio, validation_ratio, train_ratio)!')
    elif not np.all(np.asarray(ratio) >= 1):
        raise ValueError(f'The ratio must in range [0, 1] ! Current value: {ratio=}')
    elif not np.all(int_check):  # 检查数值类型
        raise ValueError(f'The ratio value must be an integer ! Current value: {ratio=}')

    # annotation_txts 是一个文件夹，用于存放转换后的 txt 标注文件
    annotation_txts = images_jsons_folder / 'txts'
    # 把 json 标注文件转换为 txt 格式的标签文件
    if annotation_txts.exists() and overwrite_annotations:
        shutil.rmtree(annotation_txts)
    annotation_txts.mkdir()
    # 把 json 标注文件转换为 txt 标签文件。
    get_polygons_for_all_jsons(path_jsons=raw_jsons,
                               rectangle_diagonal=True, path_txts=annotation_txts)
    if images_labels_folder is not None:
        images_labels_folder = pathlib.Path(images_labels_folder).expanduser().resolve()
        if not images_labels_folder.exists():
            raise FileNotFoundError(f'The folder does not exist: {images_labels_folder}')
        # 新建 images_folder 和 labels_folder 文件夹。
        images_folder = images_labels_folder / 'images'
        labels_folder = images_labels_folder / 'labels'
    else:
        # 新建 images_folder 和 labels_folder 文件夹，放在和 images_jsons_folder 同级的位置。
        images_folder = images_jsons_folder.parent / 'images'
        labels_folder = images_jsons_folder.parent / 'labels'

    images_train = images_folder / 'train'
    images_validation = images_folder / 'validation'
    images_test = images_folder / 'test'

    labels_train = labels_folder / 'train'
    labels_validation = labels_folder / 'validation'
    labels_test = labels_folder / 'test'

    if overwrite_images_labels:
        shutil.rmtree(images_folder)
        images_train.mkdir(parents=True)
        images_validation.mkdir()
        images_test.mkdir()

        shutil.rmtree(labels_folder)
        labels_train.mkdir(parents=True)
        labels_validation.mkdir()
        labels_test.mkdir()

    train_quantity = ratio[0]
    validation_quantity = ratio[1]
    test_quantity = ratio[2]
    print(f'Dataset split ratio: {train_quantity=}, '
          f'{validation_quantity=}, {test_quantity=}')
    # 使用 tally 等进行统计，以免出现文件缺漏的情况。
    processed = 0
    tally = 0
    copied = 0
    tqdm_raw_images = tqdm(raw_images.iterdir(),
                           total=len(os.listdir(raw_images)), ncols=80)
    for each in sorted(tqdm_raw_images):  # iterdir 默认是乱序的，所以要用 sorted 。
        # 有子文件夹时，应该对子文件夹内的图片再次排序。 img_quantity 是图片数量，用 walrus 表达式得到。
        if each.is_file():
            # 把图片对应的标签 txt 文件也放到相应文件夹
            each_annotation_name = each.stem + '.txt'
            each_annotation = annotation_txts / each_annotation_name
            if each_annotation.exists():  # 允许没有标注的图片存在，因为有时图片中确实没有物体
                copy_one_file(
                    each_annotation, processed, copied, tally, train_quantity,
                    validation_quantity, test_quantity,
                    train_dir=labels_train, validation_dir=labels_validation, test_dir=labels_test)

            # 复制了标签之后，再计算 processed, tally 等数值，保证图片和标注的 tally 等始终一样。
            processed, copied, tally = copy_one_file(
                each, processed, copied, tally, train_quantity,
                validation_quantity, test_quantity,
                images_train, images_validation, images_test)

    missing = tally - copied
    assert (missing == 0), f'Alert! Some images are not moved, quantity: {missing}'
    print(f'Images quantity: {tally}')
    print('Done!')


def main():
    """生成 txt 标签文件，并把图片和标签分到 3 个数据集中。"""
    images_jsons_folder = r'/media/drin/shared_disk/work/cv/2023_06_07_connector/' \
                          r'dataset_connector/original_data/2023-06-13 2P 光伏板接插件'
    images_labels_folder = r'/media/drin/shared_disk/work/cv/2023_06_07_connector/dataset_connector'
    extract_txt_and_split_dataset(images_jsons_folder=images_jsons_folder,
                                  images_labels_folder=images_labels_folder,
                                  images_jsons_in_one=False, overwrite_images_labels=False)


if __name__ == '__main__':
    main()


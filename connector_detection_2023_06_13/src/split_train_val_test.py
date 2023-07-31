#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于把输入文件夹内的图片，按比例复制到 3 个文件夹： train, validation, test。主要适用于
 YOLOv8 的物体检测 object detection 任务。

在分配到 3 个数据集之前，会先根据 json 标注文件，生成 txt 标签文件。
默认的 train, validation, test 分配比例是 3:1:1 。可以手动修改这个比例值。

版本号： 0.5
日期： 2023-07-18
作者： jun.liu@leapting.com
"""
import os
import pathlib
import shutil

import numpy as np
from tqdm import tqdm

import json2txt_yolov8


def copy_one_file(each, processed, tally,
                  train_quantity, validation_quantity, test_quantity,
                  train_dir, validation_dir, test_dir):
    """拷贝单个是图片或者标签。按照 :attr:`processed` 计数值，分配到 3 个训练集、验证集或测试集的文件夹中。

    Arguments:
        each (str | pathlib.Path): 一个字符串或 Path 对象，指向一个图片或是一个标签文件。
        processed (int): 一个 [0, 5] 范围内的整数，表示已经处理过的图片或标签数量。
        tally (int): 一个整数，表示总共处理过的图片或标签数量。
        train_quantity (int): 一个整数，表示训练集划分比例，通常为 3。
        validation_quantity (int): 一个整数，表示验证集的划分比例，通常为 1。
        test_quantity (int): 一个整数，表示测试集的划分比例，通常为 1。
        train_dir (str | Path): 一个字符串，指向一个文件夹，存放训练集的数据。
        validation_dir (str | Path): 一个字符串，指向一个文件夹，存放验证集的数据。
        test_dir (str | Path): 一个字符串，指向一个文件夹，存放测试集的数据。

    Returns:
        processed (int): 一个 [0, 5] 范围内的整数，表示已经处理过的图片或标签数量。
        tally (int): 一个整数，表示总共处理过的图片或标签数量。
    """
    processed += 1  # 计算处理过的数量。
    if processed <= train_quantity:
        # 为了保留 metadata，使用 copy2()。如果 src 和 dst 文件名相同，会直接覆盖同名文件。
        shutil.copy2(each, train_dir)
    elif processed <= train_quantity + validation_quantity:
        shutil.copy2(each, validation_dir)
    elif processed <= train_quantity + validation_quantity + test_quantity:
        shutil.copy2(each, test_dir)
        if processed == train_quantity + validation_quantity + test_quantity:
            processed = 0  # 注意必须在 processed = 5 时归零，重新累计。
    tally += 1  # 计算处理过的数量。
    return processed, tally


def extract_txt_and_split_dataset(images_jsons_folder, yaml_path,
                                  images_jsons_in_one=False,
                                  images_labels_folder=None,
                                  ratio=(3, 1, 1),
                                  overwrite_annotations=False, overwrite_images_labels=False):
    """把 json 标注文件转换为 txt 标签文件，然后把输入文件夹 :attr:`images_jsons_folder` 内的图片和
    txt 标签文件，按比例复制到 3 个文件夹：train, validation, test。

    Arguments:
        images_jsons_folder (str): 一个字符串，指向一个文件夹，其中放着 2 个文件夹，分别是 images 和 jsons。
        yaml_path (str)： 一个字符串，指向 YOLOv8 的 yaml 数据文件。
        images_jsons_in_one (bool): 一个布尔值，如果为 True，表示 json 和 images 是同一个文件夹。
        images_labels_folder (str): 一个字符串，指向一个文件夹，其中放着 2 个文件夹，分别是 images 和 labels。
            labels 中的文件是 txt 格式的标签文件。
        ratio (tuple[int, int, int]): 一个元祖，其中包含 3 个整数，分别表示训练集，验证集和测试集的划分比例。
        overwrite_annotations (bool): 一个布尔值，如果为 True，则会把之前的 txt 标注文件删除，然后重新生成标注文件。
        overwrite_images_labels (bool): 一个布尔值，如果为 True，则会把之前的图片和标签删除，然后重新生成。
    """
    images_jsons_folder = pathlib.Path(images_jsons_folder).expanduser().resolve()
    if images_jsons_in_one:
        raw_images = images_jsons_folder
        raw_jsons = images_jsons_folder
    else:
        raw_images = images_jsons_folder / 'images'  # images
        raw_jsons = images_jsons_folder / 'jsons'
    if not raw_images.exists():
        raise FileNotFoundError(f'The folder does not exist: {raw_images}')
    if not raw_jsons.exists():
        raise FileNotFoundError(f'The folder does not exist: {raw_jsons}')

    yaml_path = pathlib.Path(yaml_path).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f'The folder does not exist: {yaml_path}')

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
    if annotation_txts.exists():
        if overwrite_annotations:
            shutil.rmtree(annotation_txts)
    else:
        annotation_txts.mkdir()
    # 把 json 标注文件转换为 txt 标签文件。
    json2txt_yolov8.get_polygons_for_all_jsons(path_jsons=raw_jsons, yaml_path=yaml_path,
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
        shutil.rmtree(labels_folder)

    train_val_test_folders = (images_train, images_validation,
                              images_test, labels_train,
                              labels_validation, labels_test)
    for each_folder in train_val_test_folders:
        if not each_folder.exists():
            each_folder.mkdir(parents=True)

    train_quantity = ratio[0]
    validation_quantity = ratio[1]
    test_quantity = ratio[2]
    print(f'Dataset split ratio: {train_quantity=}, '
          f'{validation_quantity=}, {test_quantity=}')
    # 使用 tally 等进行统计，以免出现文件缺漏的情况。
    processed = 0
    tally = 0

    with tqdm(total=len(os.listdir(raw_images)), desc='processing images', unit=' images',
              leave=True, ncols=100) as progress_bar:
        for each in sorted(raw_images.iterdir()):  # iterdir 默认是乱序的，所以要用 sorted 。
            progress_bar.update(1)  # 更新进度条。
            if each.is_file():
                # 把图片对应的标签 txt 文件也放到相应文件夹
                each_annotation_name = each.stem + '.txt'
                each_annotation = annotation_txts / each_annotation_name
                if each_annotation.exists():  # 允许没有标注的图片存在，因为有时图片中确实没有物体
                    copy_one_file(
                        each_annotation, processed, tally, train_quantity,
                        validation_quantity, test_quantity,
                        train_dir=labels_train, validation_dir=labels_validation, test_dir=labels_test)

                # 复制了标签之后，再计算 processed, tally 等数值，保证图片和标注的 tally 等始终一样。
                processed, tally = copy_one_file(
                    each, processed, tally, train_quantity,
                    validation_quantity, test_quantity,
                    images_train, images_validation, images_test)

    print(f'Images quantity: {tally}')
    print('Done!')


def copy_images_labels(source_images_folder, source_labels_folder,
                       target_images_folder, target_labels_folder, ratio=(3, 1, 1)):
    """把 source 文件夹中的每个图片，及其对应的标签文件，从 source 文件夹复制到 target 文件夹。

    Arguments:
        source_images_folder (str): 一个字符串，指向一个文件夹，是图片的源文件夹。
        source_labels_folder (str): 一个字符串，指向一个文件夹，是标签的源文件夹。
        target_images_folder (str): 一个字符串，指向一个文件夹，是图片的目标文件夹。
        target_labels_folder (str): 一个字符串，指向一个文件夹，是标签的目标文件夹。
        ratio (tuple[int, int, int]): 一个元祖，其中包含 3 个整数，分别表示训练集，验证集和测试集的划分比例。
    """

    train_quantity = ratio[0]
    validation_quantity = ratio[1]
    test_quantity = ratio[2]
    print(f'Dataset split ratio: {train_quantity= }, '
          f'{validation_quantity= }, {test_quantity= }')

    source_images_folder = pathlib.Path(source_images_folder).expanduser().resolve()
    if not source_images_folder.exists():
        raise FileNotFoundError(f'Folder not found: {source_images_folder}')
    print(f'{source_images_folder= }')
    source_labels_folder = pathlib.Path(source_labels_folder).expanduser().resolve()
    if not source_labels_folder.exists():
        raise FileNotFoundError(f'Folder not found: {source_labels_folder}')
    print(f'{source_labels_folder= }')

    target_images_folder = pathlib.Path(target_images_folder).expanduser().resolve()
    if not target_images_folder.exists():
        raise FileNotFoundError(f'Folder not found: {target_images_folder}')
    print(f'{target_images_folder= }')
    target_labels_folder = pathlib.Path(target_labels_folder).expanduser().resolve()
    if not target_labels_folder.exists():
        raise FileNotFoundError(f'Folder not found: {target_labels_folder}')
    print(f'{target_labels_folder= }')

    target_images_train = target_images_folder / 'train'
    target_images_validation = target_images_folder / 'validation'
    target_images_test = target_images_folder / 'test'

    target_labels_train = target_labels_folder / 'train'
    target_labels_validation = target_labels_folder / 'validation'
    target_labels_test = target_labels_folder / 'test'

    # 如果上面 6 个文件夹不存在，则进行创建
    train_val_test_folders = (target_images_train, target_images_validation,
                              target_images_test, target_labels_train,
                              target_labels_validation, target_labels_test)
    for each_folder in train_val_test_folders:
        if not each_folder.exists():
            each_folder.mkdir(parents=True)

    processed = 0
    tally = 0
    with tqdm(total=len(os.listdir(source_images_folder)), desc='processing images', unit=' images',
              leave=True, ncols=100) as progress_bar:
        # iterdir 默认是乱序的，所以要用 sorted 。而排序则是因为要把图片按顺序排列，才能使得训练集和
        # 验证、测试集的分布相同。
        for each_image in sorted(source_images_folder.iterdir()):
            progress_bar.update(1)  # 更新进度条。
            if each_image.is_file():  # 跳过文件夹，只处理图片。
                # 把图片对应的标签 txt 文件也放到相应文件夹
                each_label_name = each_image.stem + '.txt'
                each_label = source_labels_folder / each_label_name
                if each_label.exists():  # 允许没有标注的图片存在，因为有时图片中确实没有物体
                    copy_one_file(
                        each_label, processed, tally,
                        train_quantity, validation_quantity, test_quantity,
                        train_dir=target_labels_train, validation_dir=target_labels_validation,
                        test_dir=target_labels_test)

                # 复制了标签之后，再计算 processed, tally 等数值，保证图片和标注的 tally 等始终一样。
                processed, tally = copy_one_file(
                    each_image, processed, tally,
                    train_quantity, validation_quantity, test_quantity,
                    train_dir=target_images_train, validation_dir=target_images_validation,
                    test_dir=target_images_test)

    print(f'Images quantity: {tally}')
    print('Done!')


def main():
    """生成 txt 标签文件，并把图片和标签分到 3 个数据集中。"""
    # 2023-06-08 白色背景接插件 2023-06-12 1P 光伏板接插件 2023-06-13 2P 光伏板接插件
    # images_jsons_folder = r'/media/disk_2/work/cv/2023_06_07_connector/dataset_connector/original_data/2023-07-13 湖州巡检车拍的放大图片'  # 1st
    # yaml_path = r'/media/disk_2/work/cv/2023_06_07_connector/tryout/src/connector.yaml'
    # images_labels_folder = r'/media/disk_2/work/cv/2023_06_07_connector/dataset_connector'
    # extract_txt_and_split_dataset(images_jsons_folder=images_jsons_folder,
    #                               yaml_path=yaml_path,
    #                               images_labels_folder=images_labels_folder,
    #                               images_jsons_in_one=False,
    #                               overwrite_annotations=False, overwrite_images_labels=False)

    source_images_folder = r'~/liuj/2023_06_07_connector/dataset_connector/original_data/2023-07-24 国宁增加的图片/img'
    source_labels_folder = r'~/liuj/2023_06_07_connector/dataset_connector/original_data/2023-07-24 国宁增加的图片/txt'

    target_images_folder = r'~/liuj/2023_06_07_connector/dataset_connector/images'
    target_labels_folder = r'~/liuj/2023_06_07_connector/dataset_connector/labels'
    copy_images_labels(source_images_folder=source_images_folder,
                       source_labels_folder=source_labels_folder,
                       target_images_folder=target_images_folder,
                       target_labels_folder=target_labels_folder, ratio=(3, 1, 1))


if __name__ == '__main__':
    main()

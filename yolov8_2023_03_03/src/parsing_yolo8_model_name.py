#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取文件夹中 YOLOv8 模型的名字，解析出 conf 和 iou 两个参数值。

用法：导入模型后调用 parsing_name 函数，返回一个字典，包含了所需要的 3 个参数。
    info = parsing_name(yolo8_folder)
    model_name = info['model_name']
    conf = info['conf']
    iou = info['iou']
要求：yolo8_folder 是一个文件夹，其中只放需要使用的最新 YOLOv8 模型，模型名字
格式为 xxx_confn1_ioun2.pt，即名字以 conf 和 iou 部分结尾。
yolo8_folder 中可以有其它子文件夹，备份的 YOLOv8 模型可以放到子文件夹中。

版本号： 0.3
作者： jun.liu@leapting.com
"""
import pathlib
import sys


def parsing_name(yolo8_folder):
    """读取文件夹中 YOLOv8 模型的名字，解析出 conf 和 iou 两个参数值。

    Arguments:
        yolo8_folder (str): 一个字符串，代表一个文件夹，其中只放需要使用的最新
          YOLOv8 模型，模型名字格式为 xxx_confn1_ioun2.pt，即名字以 conf 和 iou
          部分结尾。yolo8_folder 中可以有其它子文件夹，备份的 YOLOv8 模型可以
          放到子文件夹中。
    Returns:
        info (dict)：一个字典，其中包含了 model_name,  conf 和 iou 三个 key。
    """
    # Path.resolve 用于返回绝对路径，并去掉 ~ 和 .. 等相对路径
    model_path = pathlib.Path(yolo8_folder).expanduser().resolve()
    model_found = 0
    info = None
    for each in model_path.iterdir():
        name_wo_suffix = each.stem  # stem 属性得到去掉了后缀的文件名
        if (each.suffix in ['.pt', '.pth']) and (
                'conf' in name_wo_suffix) and ('iou' in name_wo_suffix):
            model_found += 1

            # 此时 split_name 是一个列表，包括 [before conf, after conf]。
            split_name = name_wo_suffix.split('conf')

            # split_name[1] 是 conf 后面的部分。
            split_name = split_name[1].split('_')
            # 此时 split_name 是一个列表，包括 [conf, left file name]。
            conf = float(split_name[0])
            # 此时 split_name[1] 是 iou 后面的部分。
            split_name = split_name[1].split('iou')
            iou = float(split_name[1])
            # 检查 conf 和 iou，确保数值在一个合理的范围。
            assert (1 > conf > 0) and (1 > iou > 0), (
                'Both conf and iou must in range (0, 1). Current value: '
                f'{conf=}, {iou=}')

            info = {'model_name': each.name, 'conf': conf, 'iou': iou}

    if model_found != 1:
        raise FileNotFoundError(
            'No suitable YOLOv8 model is found, or more than one model is '
            f'found. \nPlease check the folder: {model_path}')
    # 最后显示解析的结果。
    print(f'Using model: {info["model_name"]}\n'
          f'conf={info["conf"]}, iou={info["iou"]}')

    return info


if __name__ == '__main__':
    if len(sys.argv) == 2:
        info = parsing_name(sys.argv[1])
    else:
        # yolo8_folder = '../pretrained_models'
        # yolo8_folder = r'../output models/2023-04-23 x44'
        yolo8_folder = r'~/work/cv/2023_02_24_yolov8/output models/2023-04-23 x44'
        info = parsing_name(yolo8_folder)




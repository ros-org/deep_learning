#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本模块用于把 labelme 标注的 json 文件转换为 YOLOv8 需要的 txt 格式，该
txt 文件可以用于分割 instance segmentation 和检测 object detection 任务。

版本号： 0.3
日期： 2023-06-06
作者： jun.liu@leapting.com
"""
import json
import os
import pathlib
import shutil
import sys

from tqdm import tqdm


def _get_polygons_in_one_json(path_json, path_txts=None):
    """把一个 json 文件内所有的多边形标注提取出来，生成一个 txt 文件。

    Args：
        path_json： 一个字符串，是一个标注文件 json 的存放路径。
        path_txts： 一个字符串，是用于存放新生成的 txt 文件的存放路径。
    """
    try:
        with open(path_json) as f:
            annotations_raw_dict = json.load(f)
    except FileNotFoundError:
        print(f'File not found: {path_json}')

    txt_file_name = path_json.stem + ".txt"  # 新生成的 txt 文件名
    txt_file = path_txts / txt_file_name  # txt 的路径

    image_height = annotations_raw_dict['imageHeight']
    image_width = annotations_raw_dict['imageWidth']
    assert (image_width > 0) and (image_height > 0), (
        'image_height and image_width must larger than 0.')

    found_negative = False
    # 把每一个多边形的点记录到 txt_file 中。
    with open(txt_file, 'w') as f:
        # annotations_raw_dict['shapes'] 是一个列表，存放了所有分割的多边形框。
        # 而每个多边形框，则是一个字典，包含的 key 为：(['label', 'points',
        # 'group_id', 'shape_type', 'flags'])
        for i, each_annotation in enumerate(annotations_raw_dict['shapes']):
            label_name = each_annotation['label']
            # points 是一个列表，包含了一个多边形的所有点。
            points = each_annotation['points']
            if label_name == "solar_panel":
                f.write("0 ")  # 首位放入类别 ID
            else:
                raise ValueError(f"Label name = {label_name}, need a new index.")

            for point in points:
                point_x = point[0]
                point_y = point[1]

                # 因为 labelme 有时标注会出现负数，所以做一个修改，使其等于 0.
                if point_x < 0:
                    print(f'point_x is {point_x}, changing to 0.')
                    point_x = 0
                    found_negative = True
                if point_y < 0:
                    print(f'point_y is {point_y}, changing to 0.')
                    point_y = 0
                    found_negative = True

                # assert (point_x >= 0) and (point_y >= 0), (
                #     'Point coordinates must no less than 0.')
                # 获取坐标点的比例值。
                point_x_scaled = point_x / image_width
                point_y_scaled = point_y / image_height

                f.write(f'{point_x_scaled} ')
                f.write(f'{point_y_scaled} ')

            if found_negative:
                print(f'Error in file:\n{path_json}\n')
                found_negative = False
            f.write('\n')  # 每个多边形，必须以换行 \n 结束。


def _get_rectangles_in_one_json(path_json, path_txts=None):
    """把一个 json 文件内所有的矩形标注提取出来，生成一个 txt 文件。

    Args：
        path_json (str)： 一个字符串，是一个标注文件 json 的存放路径。
        path_txts (str)： 一个字符串，是用于存放新生成的 txt 文件的存放路径。
    """
    try:
        with open(path_json) as f:
            print(f'debug 1, {path_json= }')
            annotations_raw_dict = json.load(f)
    except FileNotFoundError:
        print(f'File not found: {path_json}')

    image_height = annotations_raw_dict['imageHeight']
    image_width = annotations_raw_dict['imageWidth']
    if (image_width <= 0) or (image_height <= 0):
        print(f'Error in file: {path_json}')
        raise ValueError('image_height and image_width must larger than 0.')

    txt_file_name = path_json.stem + ".txt"  # 新生成的 txt 文件名
    txt_file = path_txts / txt_file_name  # txt 的路径

    found_negative = False
    # 把每一个物体框的点记录到 txt_file 中。
    with open(txt_file, 'w') as f:
        # annotations_raw_dict['shapes'] 是一个列表，存放了所有分割的物体框。
        # 而每个多边形框，则是一个字典，包含的 key 为：(['label', 'points',
        # 'group_id', 'shape_type', 'flags'])
        for j, each_annotation in enumerate(annotations_raw_dict['shapes']):
            label_name = each_annotation['label']
            # points 是一个列表，包含了一个矩形框的左上角和右下角点。
            points = each_annotation['points']
            # TODO 后续添加代码，处理多种识别目标
            if label_name == "disconnected":  # TODO: 后续把 label_name 作为参数。 solar_panel abnormal
                f.write("0 ")  # 首位放入类别 ID
            else:
                raise ValueError(f"Label name = {label_name}, need a new index.")

            # 矩形框只有 2 个点的坐标。
            for j, point in enumerate(points):
                point_x = point[0]
                point_y = point[1]

                # 因为 labelme 有时标注会出现负数，所以做一个修改，使其等于 0.
                if point_x < 0:
                    print(f'point_x is {point_x}, changing to 0.')
                    point_x = 0
                    found_negative = True
                if point_y < 0:
                    print(f'point_y is {point_y}, changing to 0.')
                    point_y = 0
                    found_negative = True

                if j == 0:  # 左上角点
                    top_left_x = point_x
                    top_left_y = point_y
                elif j == 1:  # 右下角点
                    bottom_right_x = point_x
                    bottom_right_y = point_y

            rectangle_center_x = (bottom_right_x + top_left_x) / 2
            rectangle_center_y = (bottom_right_y + top_left_y) / 2

            # 注意高度宽度要用绝对值。因为矩形框对角点的顺序可能是左上和右下，也可能是右下和左上。
            rectangle_width = abs(bottom_right_x - top_left_x)
            rectangle_height = abs(bottom_right_y - top_left_y)
            if (rectangle_width <= 0) or (rectangle_height <= 0):
                print(f'{path_json}')
                raise ValueError(f'{rectangle_width= }, {rectangle_height= }')

            # 把数值转换为一个比例值。
            center_x_scaled = rectangle_center_x / image_width
            center_y_scaled = rectangle_center_y / image_height
            rectangle_width_scaled = rectangle_width / image_width
            rectangle_height_scaled = rectangle_height / image_height

            f.write(f'{center_x_scaled} ')  # 用空格分开各个数值。
            f.write(f'{center_y_scaled} ')
            f.write(f'{rectangle_width_scaled} ')
            f.write(f'{rectangle_height_scaled} ')

            if found_negative:
                print(f'Error in file:\n{path_json}\n')
                found_negative = False
            f.write('\n')  # 每个物体框，必须以换行 \n 结束。


def get_polygons_for_all_jsons(path_jsons, rectangle_diagonal,
                               path_txts=None, overwrite_txt=True):
    """对文件夹内每一个 json 标注文件，提取所有多边形标注，生成一个 txt 文件。

    输入：
        path_jsons： 一个字符串，是标注文件 json 的存放路径。
        rectangle_format (bool)： 一个字符串，是一个 labelme 标注或者 anylabeling 标注，
            格式为 tlbr，即 Top Left，...标注文件 json 的存放路径。

        path_txts： 一个字符串，是用于存放新生成的 txt 文件的存放路径。
    """
    path_jsons = pathlib.Path(path_jsons).expanduser().resolve()
    if not path_jsons.exists():
        raise FileNotFoundError(f'Json path is not found: {path_jsons}')

    if path_txts is None:
        path_txts = path_jsons  # 和 json 放到同一个文件夹。
    else:
        path_txts = pathlib.Path(path_txts).expanduser().resolve()

    if path_txts.exists() and overwrite_txt:
        shutil.rmtree(path_txts)
    path_txts.mkdir(parents=True)

    # 如果只输入了一个 json 文件，则直接进行提取。如果输入的是文件夹，则进行
    # 循环，逐个提取。
    if path_jsons.is_file():
        if rectangle_diagonal:
            _get_rectangles_in_one_json(path_json=path_jsons, path_txts=path_txts)
        else:
            _get_polygons_in_one_json(path_json=path_jsons, path_txts=path_txts)
    else:
        tqdm_iterator = tqdm(path_jsons.iterdir(), total=len(os.listdir(path_jsons)), ncols=80)
        for one_path in tqdm_iterator:
            if rectangle_diagonal:
                _get_rectangles_in_one_json(path_json=one_path, path_txts=path_txts)
            else:
                _get_polygons_in_one_json(path_json=one_path, path_txts=path_txts)

    print(f'Extracting txts done!')


if __name__ == '__main__':
    # 先设置输入 path_jsons 和输出文件夹 path_txts。 train validation test
    datasets = ['train', 'validation', 'test']
    for dataset in datasets:
        path_jsons = r'/home/drin/work/cv/2023_05_24_infrared/dataset/labels/' \
                     f'jsons/{dataset}'
        path_txts = r'/home/drin/work/cv/2023_05_24_infrared/dataset/labels/' \
                    f'{dataset}'

        if len(sys.argv) == 3:  # 后 2 个参数分别为 json 路径和 txt 路径。
            get_polygons_for_all_jsons(path_jsons=sys.argv[1],
                                       rectangle_diagonal=True, path_txts=sys.argv[2])  # noqa
        else:
            get_polygons_for_all_jsons(path_jsons=path_jsons,
                                       rectangle_diagonal=True, path_txts=path_txts)  # noqa



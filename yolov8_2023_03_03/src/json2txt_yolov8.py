#!/usr/bin/env python3
# -*- coding utf-8 -*-
"""本模块用于把 labelme 标注的 json 文件转换为 YOLOv8 需要的 txt 格式，并且该
txt 文件只用于 instance segmentation。
如需支持，可联系 jun.liu@leapting.com
"""

import json
from pathlib import Path
import sys


def _get_polygons_in_one_json(path_json, path_txts=None):
    """把一个 json 文件内所有的多边形标注提取出来，生成一个 txt 文件。

    输入：
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
            # 存放的是将要放入到 txt 文件的坐标点信息。程序稳定后可以删除 points_2_txt。 =========================
            points_2_txt = []
            if label_name == "solar_panel":
                points_2_txt.append(0)  # 首位放入类别 ID
                f.write("0 ")
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

                points_2_txt.append(point_x_scaled)
                points_2_txt.append(point_y_scaled)
                f.write(f'{point_x_scaled} ')
                f.write(f'{point_y_scaled} ')

            if found_negative:
                print(f'Error in file:\n{path_json}\n')
                found_negative = False
            f.write('\n')  # 每个多边形，必须以换行 \n 结束。


def get_polygons_for_all_jsons(path_jsons, path_txts=None):
    """对文件夹内每一个 json 标注文件，提取所有多边形标注，生成一个 txt 文件。

    输入：
        path_jsons： 一个字符串，是标注文件 json 的存放路径。
        path_txts： 一个字符串，是用于存放新生成的 txt 文件的存放路径。
    """

    path_jsons = Path(path_jsons)
    if path_txts is None:
        path_txts = path_jsons
    else:
        path_txts = Path(path_txts)

    if (not path_jsons.exists()) or (not path_txts.exists()):
        raise FileExistsError('One of the input paths does not exist!')

    # 如果只输入了一个 json 文件，则直接进行提取。如果输入的是文件夹，则进行
    # 循环，逐个提取。
    if path_jsons.is_file():
        _get_polygons_in_one_json(path_json=path_jsons, path_txts=path_txts)
    else:
        # 考虑用 tqdm
        for one_path in path_jsons.iterdir():
            _get_polygons_in_one_json(path_json=one_path, path_txts=path_txts)

    print(f'Extracting txts done!')


if __name__ == '__main__':
    # 先设置输入 path_jsons 和输出文件夹 path_txts。 train validation test
    path_jsons = r'../solar_panel_data/labels/jsons/test'
    path_txts = r'../solar_panel_data/labels/test'

    if len(sys.argv) == 3:  # 后 2 个参数分别为 json 路径和 txt 路径。
        get_polygons_for_all_jsons(
            path_jsons=sys.argv[1], path_txts=sys.argv[2])
    else:
        get_polygons_for_all_jsons(path_jsons=path_jsons,
                                   path_txts=path_txts)



#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""本模块用于把输入的方框，从 center_x, center_y, width, height 格式改为
top left point, bottom right point 格式。
另外一个功能是，把图片的标签 txt 文件中的方框，在图片中画出来，并保存该图片。

版本号： 0.1
日期： 2023-06-05
作者： jun.liu@leapting.com
"""
import os
import pathlib

import cv2 as cv
import numpy as np


def xywh_2_tlbr(box_xywh: tuple[float, float, float, float],
                height_width):
    """把输入的方框，从 center_x, center_y, width, height 格式改为
        top left point, bottom right point 格式。
    """
    box_center_x = box_xywh[0]
    box_center_y = box_xywh[1]
    box_width = box_xywh[2]
    box_height = box_xywh[3]
    box_top_left_x = box_center_x - box_width / 2
    box_top_left_y = box_center_y - box_height / 2
    box_top_left = int(box_top_left_x * height_width[1]), int(
        box_top_left_y * height_width[0])

    box_bottom_right_x = box_center_x + box_width / 2
    box_bottom_right_y = box_center_y + box_height / 2
    box_bottom_right = int(box_bottom_right_x * height_width[1]), int(
        box_bottom_right_y * height_width[0])
    return box_top_left, box_bottom_right


def save_image(img_path, results_dir=None):
    """把值写到对应图片中，并保存到路径 :attr:`results_dir` 中。

    Arguments:
        img_path (str): 一个 Path 对象，指向一张图片。可以是相对路径或绝对路径。
        results_dir (str): 一个 Path 对象，指向一个文件夹。最终图片将被保存到这个文件夹中。
    """
    img_path = pathlib.Path(img_path).expanduser().resolve()
    if results_dir is None:
        results_dir = img_path.parent
    else:
        results_dir = pathlib.Path(results_dir).expanduser().resolve()

    labels_name = img_path.stem + '.txt'
    labels = img_path.parent / labels_name
    print(f'{labels= }')
    boxes = []
    with open(labels, 'r') as f:
        for line in f:
            split_line = line.split()  # split 会去掉空格和换行符等
            # print(f'{len(split_line)= } {split_line= }')
            box = [round(float(num), 4) for num in split_line[1:]]
            boxes.append(box)
            class_name = split_line[0]

    print(f'{boxes= }')

    img = cv.imread(str(img_path))  # noqa， OpenCV 需要输入字符串格式
    text = f'{class_name}'
    height_width = img.shape[: 2]  # height, width
    min_size = np.amin(height_width)
    font_scale = min_size / 400
    font_face = cv.FONT_HERSHEY_SIMPLEX
    line_thickness = 1 if min_size < 1000 else 2

    # text_size 是一个元祖，包含 ((text_width, text_height), baseline)
    # text_size 返回的是文本的尺寸，用这个尺寸画框，可以恰好把文本包住。
    text_size = cv.getTextSize(text, font_face, font_scale, line_thickness)
    text_width = text_size[0][0]
    text_height = text_size[0][1]
    baseline = text_size[1] + line_thickness

    # 画一个蓝色填充框，包住整个文字
    blue_background = 255, 0, 0  # BGR 格式的颜色
    filled_rectangle = cv.FILLED
    rectangle_bottom_right = text_width, text_height + 2 * baseline
    cv.rectangle(img, (0, 0), rectangle_bottom_right,
                 blue_background, filled_rectangle)

    text_bottom_left_point = 0, text_height + baseline
    anti_aliased = cv.LINE_AA  # anti aliased line
    white = 255, 255, 255
    cv.putText(img, text, text_bottom_left_point, font_face,
               font_scale, white, line_thickness, anti_aliased)

    for each_box in boxes:
        cv.rectangle(img, *xywh_2_tlbr(each_box, height_width),  # noqa
                     blue_background, thickness=3)
    # cv.rectangle(img, *xywh_2_tlbr(box2, height_width),  # noqa
    #              blue_background, thickness=3)

    result_img = results_dir / f'{img_path.stem}_box.jpg'
    if result_img.exists():
        os.remove(result_img)  # 如果已经存在结果，则删除
    cv.imwrite(str(result_img), img)  # noqa， OpenCV 需要输入字符串格式
    print(f'Done, image saved as: {result_img}')


if __name__ == '__main__':
    # img_path = '无人机视频.mp4_000000.879.jpg'
    img_path = '../temp/正面_02_20230601180359224_008.jpg'
    results_dir = None
    save_image(img_path=img_path, results_dir=results_dir)

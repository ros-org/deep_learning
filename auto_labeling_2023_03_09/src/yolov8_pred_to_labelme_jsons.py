#!/usr/bin/env python3
# -*- coding utf-8 -*-
"""本模块用于把 YOLOv8 的预测结果，转换为 labelme 的标注，输出一个 json 文件。
版本号： 0.1
如需帮助，可联系 jun.liu@leapting.com
"""
import json
import pathlib
import sys

import cv2 as cv
import numpy as np
from ultralytics import YOLO


def _yolov8_prediction(model, one_image, object_confidence, save_predictions,
                       extract_every_points, epsilon):
    """把 YOLOv8 的预测结果，转换为 labelme 的标注，输出一个 json 文件。

    model： 一个字符串路径，指向一个训练好的 YOLOv8 实例分割模型，
        格式为 .pt。
    one_image： 一个 pathlib.Path 对象，指向一个待预测的图片。
    object_confidence： 一个范围在 [0, 1] 的浮点数，表示在预测时，使用的
        物体识别置信度阈值。
    save_predictions： 一个布尔值，如果为 True，则会把模型的预测结果也
        保存下来。
    extract_every_points： 一个整数，表示从模型预测的多边形框外边缘点中，
        每隔 extract_every_points 个点，取出一个点。
    epsilon： 一个整数，是 approxPolyDP 函数在简化多边形时，允许的最大距离
        误差（应该是使用像素点数量作为距离）。
    """
    results = model.predict(source=one_image,
                            conf=object_confidence,
                            iou=0.5,
                            save=save_predictions,
                            line_thickness=3)

    # results[0].masks 是一个张量，其中存放了预测出来的 x 个 mask，该张量
    # 形状为 [x, 416, 640]。
    mask_prediction = results[0].masks
    if mask_prediction is not None:
        # json_template 包括了 labelme 标注文件的所有项目。
        json_template = {"version": "5.1.1", "flags": {}, 'shapes': [],
                         'imagePath': str(one_image.absolute()),
                         "imageData": None,
                         'imageHeight': results[0].orig_shape[0],
                         'imageWidth': results[0].orig_shape[1]}

        # 'shapes' 是一个空列表，列表中的每一个元素来代表一个多边形框。
        polygons_list = json_template['shapes']

        # image_width_height 用于把点的坐标转换为实际值，数组形状为 (1, 2)
        # 注意是宽度在前，高度在后，因此将 orig_shape 逆序。
        image_width_height = np.array(results[0].orig_shape[::-1])

        # one_raw_polygon 是 1 个 mask 的外边框，包含了若干个点。
        # one_raw_polygon 本身是一个数组，数组形状为 (y, 2)，表示这个多边形
        # 有 y 个点组成，每个点有 2 个坐标值。
        for one_raw_polygon in mask_prediction.segments:
            # 每 extract_every_points 个点，保留一个点
            # filtered_polygon 形状为 (z, 2), z 是过滤后点的数量
            filtered_polygon = one_raw_polygon[::extract_every_points, ...]

            filtered_polygon *= image_width_height  # 把坐标从比例值转换为实际值

            # 为了使用 approxPolyDP，需要先转换为 3D 张量，增加一个维度。
            filtered_polygon = filtered_polygon[:, np.newaxis, :]
            # curve 形状必须是 (n, 1, 2) 。 epsilon 应该是表示像素点距离。
            polygon_approx = cv.approxPolyDP(  # noqa
                curve=filtered_polygon, epsilon=epsilon, closed=True)

            # 先创建一个空的字典 one_polygon，填写多边形信息之后，再放入
            # polygons_list 中。
            one_polygon = {"label": "solar_panel", "points": [],
                           "group_id": None, "shape_type": "polygon",
                           "flags": {}}  # json 中的 null，必须输入为 None

            # 只有 polygon_approx 的点数量超过 3 个时，才能形成多边形。
            if polygon_approx.shape[0] >= 3:
                # 把 polygon_approx 转换为 2D 张量，形状为 (n, 2)。
                polygon_approx = polygon_approx.squeeze()
                for point_array in polygon_approx:
                    # 从 np.float32 类型转换为 float 类型，以便 JSON 接收。
                    point = [float(each) for each in point_array]
                    # 把每个点逐个填入 one_polygon['points'] 中。
                    one_polygon['points'].append(point)

                # 把一个多边形框放入到 polygons_list 中。
                polygons_list.append(one_polygon)

        json_name = f'{one_image.stem}.json'  # 生成的 json 文件要和图片同名
        # 最后存为一个新的 json 标注文件，并且使用绝对路径。
        new_json = one_image.absolute().parent / json_name
        with open(new_json, 'w') as f:
            # 设置了 indent 之后，会自动换行，不需要设置 separators 换行符。
            json.dump(json_template, f, indent=2)
    else:
        print(f'There is no mask found in {one_image}')


def pred_to_json(trained_model=None, object_confidence=0.5,
                 save_predictions=False,
                 extract_every_points=1, epsilon=8):
    """把 YOLOv8 的预测结果，转换为 labelme 的标注，输出一个 json 文件。

    Args：
        trained_model： 一个字符串路径，指向一个训练好的 YOLOv8 实例分割模型，
            格式为 .pt。
        object_confidence： 一个范围在 [0, 1] 的浮点数，表示在预测时，使用的
            物体识别置信度阈值。
        save_predictions： 一个布尔值，如果为 True，则会把模型的预测结果也
            保存下来。
        extract_every_points： 一个整数，表示从模型预测的多边形框外边缘点中，
            每隔 extract_every_points 个点，取出一个点。
        epsilon： 一个整数，是 approxPolyDP 函数在简化多边形时，允许的最大距离
            误差（应该是使用像素点数量作为距离）。
    """
    model_and_image = r'x1_Adam_lr1e-03_b8_e100'
    if trained_model is None:
        trained_model = f'trained_models/{model_and_image}.pt'
    model = YOLO(trained_model)  # 加载训练好的 YOLOv8 模型

    # images_path = pathlib.Path(r'test_images/1_Color_42.bmp')  # 测试图片
    images_path = pathlib.Path(f'test_new_labeling/{model_and_image}')  # 测试图片文件夹
    if images_path.is_file():
        # 输入为单个文件时，单独处理。
        _yolov8_prediction(model=model, one_image=images_path,
                           object_confidence=object_confidence,
                           save_predictions=save_predictions,
                           extract_every_points=extract_every_points,
                           epsilon=epsilon)
    else:
        for one_image in images_path.iterdir():
            _yolov8_prediction(model=model, one_image=one_image,
                               object_confidence=object_confidence,
                               save_predictions=save_predictions,
                               extract_every_points=extract_every_points,
                               epsilon=epsilon)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        pred_to_json(sys.argv[1:])  # 如果是命令行，把后面几个参数传递给函数。
    else:
        pred_to_json(save_predictions=True)

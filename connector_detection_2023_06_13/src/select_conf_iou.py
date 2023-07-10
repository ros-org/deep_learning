#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""本模块用于确定最佳的 conf 和 iou 这 2 个参数。这组参数的组合，能够使得
YOLOv8 模型在测试集上，得到最大的 mAP50 指标值。
本模块适用于 YOLOv8 的检测任务 object detection。

版本号： 0.2
日期： 2023-06-13
作者： jun.liu@leapting.com
"""
import os
import pathlib
import unicodedata

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from search_lr import load_record
from search_lr import show_sorted_record
import utilities


# def show_sorted_record(map_records: pd.DataFrame, first_quantity=20):
#     map_records = map_records.sort_values(
#         by='mAP50-95', ascending=False)[: 90]
#     # 显示前 first_quantity 位。
#     # pd.set_option('display.max_rows', first_quantity)
#     np.set_printoptions(threshold=first_quantity)
#     print(f'mAP50-95: \n{map_records}\n')
#
#
# def load_record(records_name, show=True):
#     """加载保存在硬盘上的 csv 文件，文件中存放的是模型的 mAP 指标。"""
#     try:
#         map_records = pd.read_csv(records_name)
#         print(f'Using previous: {records_name} .')
#         # 用 index[-1] 找出 DataFrame 的最后一行索引。
#         counter_records = 1 + map_records.index[-1]
#
#         if show:
#             show_sorted_record(map_records)
#
#     except FileNotFoundError:
#         map_records = pd.DataFrame({})
#         counter_records = 0
#         print(f'Not found on disk: {records_name} . '
#               f'Creating a new map_records.')
#
#     return map_records, counter_records

# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:600

@utilities.timer
def main(model_path, records_name=None, sort_values='mAP50', append_record=True,
         batch_size=2, agnostic_nms=True,
         conf_range=None, conf_numbers=5,
         iou_range=None, iou_numbers=5):
    """对一个 YOLOv8 模型，寻找最佳的 conf 和 iou 参数。

    这组参数的组合，使得模型能够在测试集上，得到最大的 mask mAP(50-95) 指标值。
    Arguments:
        model_path (str): 一个路径，指向一个 YOLOv8 模型。
        records_name (str): 一个路径，指向一个 YOLOv8 模型。
        conf_range (tuple[float, float]): 一个元祖，包含 2 个浮点数，表示置信度
            conf 的范围。
        conf_numbers (int): 一个整数，表示将 :attr:`conf_range` 平均分为
            conf_numbers 个数字。
        iou_range (tuple[float, float]): 一个元祖，包含 2 个浮点数，表示置信度
            iou 的范围。
        iou_numbers (int): 一个整数，表示将 :attr:`iou_range` 平均分为
            iou_numbers 个数字。
        split (str): 一个字符串，是 'test' 或 'validation'，表示使用测试集
            或是验证集数据，对模型进行测试。
    """
    # 注意 memory_summary 必须单独打印，不能放在 f-string 中，否则显示的格式会出错。
    # print(f'debug 0, torch.cuda.memory_summary()=\n', torch.cuda.memory_summary())
    model_path = pathlib.Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    print(f'{model_path= }')
    model = YOLO(model_path)

    if records_name is None:
        model_name = model_path.stem.split('_')[0]  # 取名字第一个下划线的前面部分即可
        # 如果有事先保存好的 map_records，可以直接读取。
        records_name = f'select_conf_iou/{model_name}_records.csv'
    sort_values_test = f'{sort_values}_test'  # 因为有 2 中指标，所以按测试集指标进行排序
    map_records, counter_records = load_record(
        records_name, show=True, sort_values=sort_values_test, append_record=append_record)

    device = 0 if torch.cuda.is_available() else 'cpu'  # noqa

    total = conf_numbers * iou_numbers
    # 搜索 conf 和 iou 的范围，记录所有的 mask mAP50-95 指标。
    conf_list = np.linspace(*conf_range, num=conf_numbers)
    iou_list = np.linspace(*iou_range, num=iou_numbers)
    bouquet = unicodedata.lookup('bouquet')
    for conf in conf_list:
        for iou in iou_list:
            print(f'{bouquet}'*5, f' counting down: {total} ', '='*20,
                  f'{bouquet}'*5)
            total -= 1
            print(f'conf: {conf:.3f}, iou: {iou:.3f}.')

            metrics_test = model.val(device=device, batch=batch_size,
                                     split='test', conf=conf, iou=iou,
                                     agnostic_nms=agnostic_nms,
                                     cache=True,  # val 中使用 cache，似乎并不会加快速度？
                                     plots=False, save=False)

            metrics_val = model.val(device=device, batch=batch_size,
                                    split='val', conf=conf, iou=iou,
                                    agnostic_nms=agnostic_nms,
                                    cache=True,  # val 中使用 cache，似乎并不会加快速度？
                                    plots=False, save=False)

            # mean_results 方法会返回一个列表，包含全部的 8 个指标值。指标分成 2 组，
            # 前 4 个是 box 的指标，后 4 个是 mask 的指标。并且 4 个指标的顺序和模型
            # 输出的显示结果一一对应，即分别是 P，R，mAP50， mAP50-95 。
            # print(f'results.mean_results: {type(results.mean_results)}\n{results.mean_results()}')  # noqa
            # metrics = [round(metric, 4) for metric in results.mean_results()]

            map_records.loc[counter_records, 'conf'] = round(conf, 3)
            map_records.loc[counter_records, 'iou'] = round(iou, 3)
            map_records.loc[counter_records, 'mAP50_test'] = round(metrics_test.box.map50, 3)
            # noinspection PyTypeChecker
            map_records.loc[counter_records, 'mAP50_mean'] = round(
                np.mean((metrics_test.box.map50, metrics_val.box.map50)), 3)
            map_records.loc[counter_records, 'mAP50_val'] = round(metrics_val.box.map50, 3)
            map_records.loc[counter_records, 'precision_test'] = round(metrics_test.box.mp, 3)
            map_records.loc[counter_records, 'precision_val'] = round(metrics_val.box.mp, 3)
            map_records.loc[counter_records, 'recall_test'] = round(metrics_test.box.mr, 3)
            map_records.loc[counter_records, 'recall_val'] = round(metrics_val.box.mr, 3)

            counter_records += 1

    # 显示结果并保存。
    show_sorted_record(map_records, sort_values=sort_values_test)
    # 保存为 csv。注意设置 index=False，否则 csv 中会多出来一列索引。
    map_records.to_csv(records_name, index=False)


@utilities.timer
def predict_on_conf_iou(model_path=None, conf=None, iou=None,
                        show_boxes=False, save_txt=False, save_conf=False,
                        tryout_images=None):
    """使用 YOLOv8 模型，对指定文件夹里的图片进行分割。

    Arguments:
        model_path (str | pathlib.Path): 一个路径，指向一个 YOLOv8 模型。
        conf (float): 一个元祖，包含 2 个浮点数，表示置信度 conf。
        iou (float): 一个元祖，包含 2 个浮点数，表示置信度 iou。
        show_boxes: 一个布尔值，如果为 True，则会在分割结果的图片中，
            把分割的矩形框也显示出来。
        save_txt: 一个布尔值，如果为 True，则会生成一个 txt 文件，并把所有的
            （未经过 NMS 处理的）分割点坐标保存在这个 txt 文件中。
        save_conf: 一个布尔值，如果为 True，并且 save_txt 也为 True，则会生成
           一个 txt 文件，并把置信度 conf 保存在这个 txt 文件中。
        tryout_images (str | pathlib.Path): 一个路径，指向待分割图片的文件夹。
    """
    if model_path is None:
        # x31_Adam_lr1e-03_b4_e1200_degrees90_nbs64_close_mosaic100
        model_path = r'/home/leapting/work/cv/2023_02_24_yolov8/' \
                             r'auto_labeling/trained_models/' \
                             r'x33_Adam_lr1e-03_b4_e1200_degrees90_nbs64_close_mosaic100_hsv_v0.4.pt'  # noqa

    print(f'Using model: {pathlib.Path(model_path).name}')
    model = YOLO(model_path)

    if tryout_images is None:
        tryout_images = r'../error_checking'

    device = torch.device(0) if torch.cuda.is_available() else torch.device(
        'cpu')  # noqa
    print(f'using device: {device}')

    results = model.predict(source=tryout_images,
                            save=True,
                            device=device,
                            conf=conf,
                            iou=iou,
                            boxes=show_boxes,
                            save_conf=save_conf,
                            save_txt=save_txt,
                            line_thickness=2,
                            # retina_masks=True,  # 设置 mask 和原图大小一致。
                            workers=8,  # workers 似乎不起作用
                            )


if __name__ == '__main__':
    model_path = r'/media/disk_2/work/cv/2023_06_07_connector/pretrained_models/' \
                 r'dx020_Adam_lr1.08e-05_lrf0.01_conf0.5_iou0.7_b2_e800_nbs2_imgsz1088_hsv-h0.8-s0.8-v0.8_val954_test939.pt'  # noqa
    main(model_path=model_path,
         # sort_values='precision',
         append_record=True,
         batch_size=2, agnostic_nms=True,
         conf_range=(0.01, 0.09), conf_numbers=9,
         iou_range=(0.7, 0.2), iou_numbers=1)
    #

    # tryout_images = r'/home/leapting/work/cv/2023_02_24_yolov8/' \
    #                   r'2023_03_08_labeling_solar_panel_black_portion_only/' \
    #                   r'original images source/6. 2023-04-24 法兰泰克上采集的夜晚图片/pic20230424'  # noqa
    # tryout_images = r'/home/leapting/work/cv/2023_02_24_yolov8/' \
    #                 r'2023_03_08_labeling_solar_panel_black_portion_only/' \
    #                 r'images/test'
    # tryout_images = r'../error_checking'
    #
    # predict_on_conf_iou(model_path=model_path, conf=0.8, iou=0.75,  # conf=0.8, iou=0.75
    #                     # save_conf=True, save_txt=True,
    #                     # show_boxes=True,
    #                     tryout_images=tryout_images)

    # records_name = 'map_records.csv'
    # show_loaded_record(records_name, show=True, save=False)


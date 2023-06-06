#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""本模块用于确定最佳的 conf 和 iou 这 2 个参数。这组参数的组合，能够使得
YOLOv8 模型在测试集上，得到最大的 mask mAP(50-95) 指标值。

版本号： 0.2
日期： 2023-06-06
作者： jun.liu@leapting.com
"""
import os
import pathlib
import shutil
import time
import unicodedata

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

import utilities


def show_sorted_record(lr_records: pd.DataFrame, sort_values: str, first_quantity=20):
    lr_records = lr_records.sort_values(
        by=sort_values, ascending=False)[: 90]
    # 显示前 first_quantity 位。
    # pd.set_option('display.max_rows', first_quantity)
    np.set_printoptions(threshold=first_quantity)
    print(f'validation {sort_values}: \n{lr_records}\n')


def load_record(records_name, show=True, sort_values=None, append_record=True):
    """加载保存在硬盘上的 csv 文件，文件中存放的是模型的 mAP 指标。"""
    records_file = pathlib.Path(records_name).expanduser().resolve()
    if not records_file.parent.exists():
        records_file.parent.mkdir()  # 先创建存放 records_file 的文件夹。

    if records_file.exists():
        if append_record:
            print(f'Using previous: {records_file} .')

            lr_records = pd.read_csv(records_file)
            # 用 index[-1] 找出 DataFrame 的最后一行索引。
            counter_records = 1 + lr_records.index[-1]
            if show:
                show_sorted_record(lr_records, sort_values=sort_values)
        else:
            os.remove(records_file)
            lr_records = pd.DataFrame({})
            counter_records = 0
    else:
        print(f'Not found on disk: {records_file} . '
              f'Creating a new lr_records.')
        lr_records = pd.DataFrame({})
        counter_records = 0

    return lr_records, counter_records


@utilities.timer
def main(detect_data, batch_size=8, optimizer='SGD',
         model_type='x',
         records_name=None,
         epochs=5,
         lr_range=None, lr_numbers=5, linspace=True,
         lrf=1, imgsz=640,
         val_split='val', conf=0.5, iou=0.7,
         sort_values='mAP75', append_record=False,
         ):
    """创建一个 RegNet 模型，并且对其进行训练。 :class:`~torch.utils.data.DataLoader`

    Arguments:
        detect_data (str): 一个字符串，指向一个文件夹，其中放着训练集，验证集和测试集 3 个文件夹。
        lr (float): 一个浮点数，是模型的学习率。
        batch_size (int): 一个整数，是批量的大小。
        epochs (int): 一个整数，是训练的迭代次数。
        classes (int): 一个整数，是模型需要区分的类别数量。目前为 2 类，即只区分有异物和无异物。
        model_path (str): 一个字符串，指向一个文件夹，训练好的最佳模型将存放在此路径中。
        val_split (str): 一个字符串，val 可以使用 val 和 test 数据集。。
        save_model (bool): 布尔值。如果为 True，则会在训练过程中，把验证集准确度最高的模型保存下来。
    """
    detect_data = pathlib.Path(detect_data).expanduser().resolve()
    print(f'{detect_data= }')
    model_path = f'../pretrained_models/yolov8{model_type}.pt'
    model_path = pathlib.Path(model_path).expanduser().resolve()
    print(f'{model_path= }')

    device = 0 if torch.cuda.is_available() else 'cpu'  # noqa
    print(f'using device: {device}')

    # 时间格式 '0512'，为日期
    time_suffix = time.strftime('%m%d')
    # 如果有事先保存好的 lr_records，可以直接读取。
    records_name = f'search_lr/{records_name}_{time_suffix}.csv'
    lr_records, counter_records = load_record(
        records_name, show=True, sort_values=sort_values, append_record=append_record)

    total = lr_numbers
    # 搜索 lr 的范围，记录所有的 mask mAP50-95 指标。
    if linspace:
        lr_list = np.linspace(*lr_range, num=lr_numbers)
    else:
        lr_list = np.logspace(*np.log10(lr_range), num=lr_numbers)

    # YOLOv8 的 train 函数只接受 float 类型，所以要把 numpy 的数据类型进行一次转换
    lr_list = [float(lr) for lr in lr_list]

    bouquet = unicodedata.lookup('bouquet')
    for lr in lr_list:
        print(f'{bouquet}'*5, f' counting down: {total} ', '='*20,
              f'{bouquet}'*5)
        total -= 1

        model = YOLO(model_path)  # 每次搜索，应该新建一个模型。

        # 如果显存不够大，batch_size 应该用 8 或者更小。 plots=True, show=True
        # 如果使用 Adam 优化器，学习率至少要小于 1e-3 。
        # 这个 batch 是训练的 batch。验证的 batch 大小不受影响，似乎是默认值 16。

        model.train(
            data=str(detect_data),  patience=0,  # 注意 data 参数必须输入字符串。
            optimizer=optimizer,
            lr0=lr,
            lrf=lrf,  # 设置 lrf 为 1， 搜索学习率时，不使用学习率衰减。
            epochs=epochs, batch=batch_size, nbs=batch_size,
            device=device,
            imgsz=imgsz, plots=False,
            val=True,  # 搜索超参时，不需要看验证集的指标，所以设为 False。
            cache=True,  # 使用 cache 以加快速度。
            # amp=False,  # 不进行 AMP 检测。
            workers=8,
        )

        metrics = model.val(split=val_split, batch=batch_size, plots=False,
                            conf=conf, iou=iou)
        # print(f'{type(metrics)= }, {metrics.box.map= }')
        # print(f'{metrics.box.map50= }, {metrics.box.maps= }')

        # 物体检测的 val 方法会返回一个类 DetMetrics，包含指标和保存路径等信息。
        # metrics.box.map 是一个浮点数，即 mAP50-95。
        # metrics.box.maps 是一个数组，包含各个类别的 mAP50-95。
        # metrics.box.map50 是一个浮点数，是各个类别的 mAP50。注意目前只能获取 map50 和 map75，
        # 没有其它百分位的 map，即没有 map95 等。

        # 输出的显示结果一一对应，即分别是 P，R，mAP50， mAP50-95 。
        # print(f'results.mean_results: {type(results.mean_results)}\n{results.mean_results()}')  # noqa
        # metrics = [round(metric, 4) for metric in results.mean_results()]
        #
        lr_records.loc[counter_records, 'lr'] = f'{lr:.2e}'
        lr_records.loc[counter_records, 'mAP75'] = round(metrics.box.map75, 3)
        lr_records.loc[counter_records, 'mAP50-95'] = round(metrics.box.map, 3)
        lr_records.loc[counter_records, 'mAP50'] = round(metrics.box.map50, 3)
        lr_records.loc[counter_records, 'epochs'] = epochs
        lr_records.loc[counter_records, 'model_type'] = model_type
        lr_records.loc[counter_records, 'imgsz'] = imgsz
        lr_records.loc[counter_records, 'optimizer'] = optimizer
        counter_records += 1

    # 显示结果并保存。
    show_sorted_record(lr_records, sort_values=sort_values)
    # 保存为 csv。注意设置 index=False，否则 csv 中会多出来一列索引。
    lr_records.to_csv(records_name, index=False)


@utilities.timer
def predict_on_conf_iou(model_path=None, conf=None, iou=None,
                        show_boxes=False, save_txt=False, save_conf=False,
                        tryout_images=None, task='predict'):
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

    tryout_images = pathlib.Path(tryout_images).expanduser().resolve()
    if tryout_images is None:
        tryout_images = r'../error_checking'

    device = torch.device(0) if torch.cuda.is_available() else torch.device(
        'cpu')  # noqa
    print(f'using device: {device}')
    if task == 'predict':
        results = model.predict(source=tryout_images,
                                save=True,
                                device=device,
                                conf=conf,
                                iou=iou,
                                boxes=show_boxes,
                                save_conf=save_conf,
                                save_txt=save_txt,
                                line_thickness=2,
                                workers=8,  # workers 似乎不起作用
                                )
    elif task == 'validation':

        metrics = model.val(split='val', plots=False,
                            conf=conf, iou=iou)
        print(f'map = {round(metrics.box.map, 3)}')
        print(f'map50 = {round(metrics.box.map50, 3)}')
        print(f'map75 = {round(metrics.box.map75, 3)}')


if __name__ == '__main__':
    # x33_Adam_lr1e-03_b4_e1200_degrees90_nbs64_close_mosaic100_conf0.62_iou0.7
    # model_path = r'/home/leapting/work/cv/2023_02_24_yolov8/' \
    #              r'auto_labeling/trained_models/' \
    #              r'x44_Adam_lr1e-04_b4_e1200_degrees90_nbs64_close_mosaic100_conf0.5_iou0.79.pt'  # noqa
    detect_data = r'~/work/cv/2023_05_24_infrared/tryout/infrared.yaml'

    # main(detect_data=detect_data, records_name='lr_records', val_split='val',
    #      model_type='x', imgsz=640,
    #      optimizer='Adam', batch_size=4, epochs=5,
    #      lr_range=(3.5e-4, 3.88e-4), lr_numbers=1, linspace=True,
    #      append_record=True)
    #

    tryout_images = r'~/work/cv/2023_05_24_infrared/dataset_solar_panel/images/test'
    # tryout_images = r'../error_checking'
    model_path = r'runs/best.pt'
    predict_on_conf_iou(model_path=model_path, conf=0.5, iou=0.7,  # conf=0.8, iou=0.75
                        # save_conf=True, save_txt=True,
                        # show_boxes=True,
                        tryout_images=tryout_images,
                        # task='validation',
                        )

    # records_name = 'map_records.csv'
    # show_loaded_record(records_name, show=True, save=False)


#! /usr/bin/env python3
# -*- coding utf-8 -*-
"""本模块用于确定最佳的 conf 和 iou 这 2 个参数。这组参数的组合，能够使得
YOLOv8 模型在测试集上，得到最大的 mask mAP(50-95) 指标值。
版本号： 0.1
如需帮助，可联系 jun.liu@leapting.com
"""
import pathlib
import time
import unicodedata

import numpy as np
import pandas as pd
import torch

import utilities

from foreign_matter_classification import ForeignMatterClassifier
from foreign_matter_classification import get_data
from foreign_matter_classification import test
from foreign_matter_classification import train


def show_sorted_record(lr_records: pd.DataFrame, first_quantity=20):
    lr_records = lr_records.sort_values(
        by='val_accuracy', ascending=False)[: 90]
    # 显示前 first_quantity 位。
    # pd.set_option('display.max_rows', first_quantity)
    np.set_printoptions(threshold=first_quantity)
    print(f'validation accuracy: \n{lr_records}\n')


def load_record(records_name, show=True):
    """加载保存在硬盘上的 csv 文件，文件中存放的是模型的 mAP 指标。"""
    try:
        lr_records = pd.read_csv(records_name)
        print(f'Using previous: {records_name} .')
        # 用 index[-1] 找出 DataFrame 的最后一行索引。
        counter_records = 1 + lr_records.index[-1]

        if show:
            show_sorted_record(lr_records)

    except FileNotFoundError:
        lr_records = pd.DataFrame({})
        counter_records = 0
        print(f'Not found on disk: {records_name} . '
              f'Creating a new lr_records.')

    return lr_records, counter_records


@utilities.timer
def main(data_root, batch_size=8,
         records_name=None,
         epochs=5,
         lr_range=None, lr_numbers=5, linspace=True,
         classes=2):
    """创建一个 RegNet 模型，并且对其进行训练。

    Arguments:
        data_root (str): 一个字符串，指向一个文件夹，其中放着训练集，验证集和测试集 3 个文件夹。
        lr (float): 一个浮点数，是模型的学习率。
        batch_size (int): 一个整数，是批量的大小。
        epochs (int): 一个整数，是训练的迭代次数。
        classes (int): 一个整数，是模型需要区分的类别数量。目前为 2 类，即只区分有异物和无异物。
        model_path (str): 一个字符串，指向一个文件夹，训练好的最佳模型将存放在此路径中。
        save_model (bool): 布尔值。如果为 True，则会在训练过程中，把验证集准确度最高的模型保存下来。
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create data loaders.
    train_dataloader, validation_dataloader, test_dataloader = get_data(data_root, batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    loss_fn = torch.nn.CrossEntropyLoss()

    # 时间格式 '20230512_1016'，前半部分为日期，后半部分为小时和分钟
    time_suffix = time.strftime('%Y%m%d_%H%M')
    # 如果有事先保存好的 lr_records，可以直接读取。
    records_name = f'search_lr/{records_name}_{time_suffix}.csv'
    lr_records, counter_records = load_record(records_name, show=True)

    total = lr_numbers
    # 搜索 conf 和 iou 的范围，记录所有的 mask mAP50-95 指标。
    if linspace:
        lr_list = np.linspace(*lr_range, num=lr_numbers)
    else:
        lr_list = np.logspace(*lr_range, num=lr_numbers)
    bouquet = unicodedata.lookup('bouquet')
    for lr in lr_list:
        print(f'{bouquet}'*5, f' counting down: {total} ', '='*20,
              f'{bouquet}'*5)
        total -= 1
        print(f'lr: {lr:.2e} .')
        highest_accuracy = 0
        model = ForeignMatterClassifier(classes).to(device)  # 注意把模型移到 device 上
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        for t in range(epochs):
            print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            current_accuracy = test(validation_dataloader, model, loss_fn)
            if current_accuracy > highest_accuracy:
                highest_accuracy = current_accuracy

        lr_records.loc[counter_records, 'lr'] = f'{lr:.2e}'
        lr_records.loc[counter_records, 'val_accuracy'] = round(highest_accuracy, 3)
        counter_records += 1

    # 显示结果并保存。
    show_sorted_record(lr_records)
    # 保存为 csv。注意设置 index=False，否则 csv 中会多出来一列索引。
    lr_records.to_csv(records_name, index=False)


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
    # x33_Adam_lr1e-03_b4_e1200_degrees90_nbs64_close_mosaic100_conf0.62_iou0.7
    # model_path = r'/home/leapting/work/cv/2023_02_24_yolov8/' \
    #              r'auto_labeling/trained_models/' \
    #              r'x44_Adam_lr1e-04_b4_e1200_degrees90_nbs64_close_mosaic100_conf0.5_iou0.79.pt'  # noqa
    data_root = '~/work/cv/2023_05_04_regnet/classification_data'

    main(data_root=data_root, records_name='lr_records',
         lr_range=(2e-2, 8e-3), linspace=True,
         # lr_range=(np.log10(0.08), np.log10(0.002)), linspace=False,
         lr_numbers=5, epochs=5)
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


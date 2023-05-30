#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于训练一个 YOLOv8 检测模型，使用光伏板红外图片进行检测。例如用于检测损坏的光伏板等。

版本号： 0.1
日期： 2023-05-30
作者： jun.liu@leapting.com
"""
import pathlib

import torch
from ultralytics import YOLO

import utilities


@utilities.timer
def train_on_solar_panel(configurations=None):
    """使用光伏板数据，训练 YOLOv8 的 instance segmentation 模型。

    Arguments：
        configurations (dict): 一个字典，包含了部分需要修改的设置参数。
    """
    if configurations is None:
        configurations = {}
    model_path = configurations.get('model_path',
                                    r'../pretrained_models/yolov8x.pt')
    model_path = pathlib.Path(model_path).expanduser().resolve()
    print(f'{model_path= }')

    detect_data = configurations.get(
        'detect_data', r'~/work/cv/2023_05_24_infrared/tryout/infrared.yaml')
    detect_data = pathlib.Path(detect_data).expanduser().resolve()

    # 注意 YOLOv8.0.53 版本有问题，下面 device 的写法，在 train 和 predict 不同，
    # 要从 torch.device 变为字符串
    device = 0 if torch.cuda.is_available() else 'cpu'  # noqa
    print(f'using device: {device}')
    if device != 0:
        raise ValueError('Not using GPU! Please check the CUDA status.')

    model_id = configurations.get('model_id', r'dx01')  # n114 x36
    optimizer = configurations.get('optimizer', 'Adam')  # n114 x36
    lr0 = configurations.get('lr0', 1.1e-3)   # 1e-3  验证 1e-4 效果不好。

    lrf = configurations.get('lrf', 0.01)   # 默认 1e-2 效果不好。

    # 图片超过 1000 张时，batch 越小则速度越快，对 mAP 影响不大。
    batch = configurations.get('batch', 2)  # n114 x36

    # nominal batch size，即训练数量达到 nbs，才会有一次梯度下降。
    nbs = batch   # 验证 nbs 64 的效果不错。
    # degrees = 90  # 验证 90 有效。
    # perspective = 1e-4  # 视角变换。验证 1e-4 效果不佳。似乎可以训练更多 epochs
    # shear = 5  # 验证 5, 10, 45, 90 的效果不佳。
    # rect = True  # 把图片缩放为矩形，再进行训练。效果不佳。
    # copy_paste = 0.05  # 验证 0.05, 0.1, 0.2, 0.5 无效。
    # iou = 0.7  # 验证默认的 0.7 最佳。
    # cos_lr = True
    # flipud = 0.8  # 验证 0.8 的效果不佳。
    # translate = 0.45  # 0.1
    # mixup 是把多张图片合成，类似 cv2.addWeighted()
    # mixup = 0.2  # 验证 0.2, 0.5 的效果不佳。
    # hsv_h = 0.5  # hue 为颜色，验证 0.5 无效。
    # hsv_s = 0.3  # saturation 为饱和度。验证 0.3 无效。
    # hsv_v = 0.3  # value 为亮度。验证有晚上的场景时， 0.2, 0.9 无效。
    # fl_gamma = 2  #  验证 1.5, 2  无效。
    # 考虑加大 dfl loss gain，即加强 mask 部分的损失。因为目前 box 的 mAP 要好
    # 于 mask 的 mAP 。默认分类和 box 损失总和为 8，dfl 为 1.5，考虑加大为 7.5
    # box = 1.5  #  box 损失增益 gain。 验证 1.5 无效。
    # dfl = 1.0  #  验证 1 无效。
    epochs = configurations.get('epochs', 6)

    # close_mosaic 设置的是最后倒数若干个 epochs，关闭 mosaic。
    # 验证 50-200（epochs 为 200） 有效果。
    # close_mosaic = 100   # 验证 epochs 300，配合 close 100 效果不错。
    # momentum = 0.9  #  验证 0.95， 0.9无效。
    # scale = 0.2  #  验证 0.2, 0.8 无效。
    # mask_ratio = 2  # 对 mask 进行下采样，验证 1, 2, 10 效果均不佳。
    # imgsz = 960  # 使用大的 size
    # weight_decay = 0.05

    experiment_name = f'{model_id}_{optimizer}_lr{lr0:.2e}_b{batch}_e{epochs}_' \
                      f'nbs{nbs}'  # \
                      # f'hsv_v{hsv_v}_'_close_mosaic{close_mosaic}_degrees{degrees}

    model = YOLO(model_path)  # model_path
    # 这个 batch 是训练的 batch。验证的 batch 大小不受影响，似乎是默认值 16。

    model.train(
        data=str(detect_data), patience=0,  # 注意 data 参数必须输入字符串。
        name=experiment_name,
        optimizer=optimizer, lr0=lr0,
        lrf=lrf,
        # momentum=momentum,
        # scale=scale,
        # mask_ratio=mask_ratio,
        epochs=epochs,
        # close_mosaic=close_mosaic,
        batch=batch,
        # degrees=degrees,
        # nbs=nbs,
        # perspective=perspective,
        # shear=shear,
        # rect=rect,
        # copy_paste=copy_paste,
        # iou=iou,
        # cos_lr=cos_lr,
        # flipud=flipud,
        # translate=translate,
        # hsv_h=hsv_h,
        # hsv_s=hsv_s,
        # hsv_v=hsv_v,
        # fl_gamma=fl_gamma,
        # box=box,
        # mixup=mixup,
        # imgsz=imgsz,
        # dfl=dfl,
        # weight_decay=weight_decay,
        device=device,
        workers=8,  # 本机有 16 个处理器。
        # resume 似乎是指训练中途被打断的情况。例如设定 35 次，训练到 30 次被
        # 打断，然后再使用 resume，则会继续后面的 5 次。
        # resume=True,
        # pretrained=True,  # 似乎不起作用，可能和 YOLOv8 版本有关。
        cache=True,
    )

    # 验证训练好的模型，用的是 validation 文件夹的数据，并且 batch 大小也是
    # 上面训练时设置好的，无须再设置。但是没有 model.test() 方法。
    # 检查测试集时，输入参数 split='test'。 iou 使用和训练时相同的设置。
    metrics = model.val(split='val')  # conf=0.5, iou=0.7
    metrics = model.val(split='test')  # conf=0.5, iou=0.7

    # print(f'{type(metrics)= }, \n{metrics= }')
    # print(f', {metrics.box.map= :.2e}')
    # print(f'{metrics.box.map50= :.6f}, {metrics.box.map75= :.6f}')


@utilities.timer
def get_prediction_result(conf=0.5, iou=0.7):
    """查看预测的输出结果。"""
    model_detection = r'../pretrained_models/' \
                      r'dx01_Adam_lr1.00e-05_b4_e10_nbs4_test922.pt'  # noqa

    model = YOLO(model_detection)  # 加载训练好的模型

    tryout_images = r'random_test'
    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')  # noqa
    print(f'using device: {device}')

    model.predict(source=tryout_images,
                  save=True,
                  device=device,
                  conf=conf,
                  iou=iou,
                  # line_width=3
                  # workers=8,  # workers 似乎不起作用
                  # half=True, # half 似乎不起作用
                  )


if __name__ == '__main__':
    configurations = {'model_id': r'dx02',
                      # 'model_path': None,
                      # 'detect_data': 'coco128.yaml',
                      'batch': 4,
                      'optimizer': 'Adam',
                      'lr0': 1e-5,  # 'lr0': sgd--3.87e-3,
                      # 'lrf': 0.01,  # 'lrf': 1
                      'epochs': 10}
    # train_on_solar_panel(configurations=configurations)  # 使用现有模型，继续训练

    get_prediction_result(conf=0.5, iou=0.7)  # 0.71 0.7。


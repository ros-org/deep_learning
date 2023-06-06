#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于训练一个 YOLOv8 检测模型，使用光伏板红外图片进行检测。例如用于检测损坏的光伏板等。

版本号： 0.1
日期： 2023-05-30
作者： jun.liu@leapting.com
"""
import pathlib

import cv2 as cv
import torch
from ultralytics import YOLO

import utilities


@utilities.timer
def train_on_solar_panel(configurations=None, rename_model=True):
    """使用光伏板数据，训练 YOLOv8 的 instance segmentation 模型。

    Arguments：
        configurations (dict): 一个字典，包含了部分需要修改的设置参数。
    """
    if configurations is None:
        configurations = {}
    model_path = configurations.get(
        'model_path', r'~/work/cv/2023_05_24_infrared/pretrained_models/yolov8x.pt')
    model_path = pathlib.Path(model_path).expanduser().resolve()
    print(f'{model_path= }')

    detect_data = configurations.get(
        'detect_data', r'~/work/cv/2023_05_24_infrared/tryout/src/infrared.yaml')
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
    nbs = configurations.get('nbs', batch)  # 验证分割时， nbs 64 的效果不错。

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
                      f'nbs{nbs}'  # \_deg{degrees}
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
        nbs=nbs,
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

    # 加载训练好的最佳模型，检查其 mAP50 指标。
    best_model_path = f'runs/detect/{experiment_name}/weights/best.pt'  # 模型的默认保存路径。
    best_model_path = pathlib.Path(best_model_path).expanduser().resolve()
    best_model = YOLO(best_model_path)

    metrics_validation = best_model.val(split='val')  # conf=0.5, iou=0.7
    metrics_test = best_model.val(split='test')  # conf=0.5, iou=0.7
    map_val = round(metrics_validation.box.map, 3)
    map_test = round(metrics_test.box.map, 3)
    if rename_model:
        # 在模型名字中加上 map 指标。
        new_name = best_model_path.parent / (
            f'{experiment_name}_val{map_val * 1000:03.0f}_test{map_test * 1000:03.0f}.pt')
        best_model_path.rename(new_name)
        print(f'Best model is saved as: {new_name}\n')


@utilities.timer
def get_prediction_result(model_path, tryout_images, conf=0.5, iou=0.7, save=True):
    """查看预测的输出结果。"""
    model_path = pathlib.Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    print(f'{model_path= }')

    model = YOLO(model_path)  # 加载训练好的模型

    tryout_images = pathlib.Path(tryout_images).expanduser().resolve()
    if not tryout_images.exists():
        raise FileNotFoundError(f'Images not found: {tryout_images}')
    # one_image_path = r'random_test/无人机视频.mp4_000000.216.jpg'
    # one_image = cv.imread(one_image_path)  # noqa， OpenCV 需要输入字符串格式
    # print(f'{type(one_image)= }, {one_image.shape= }')

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')  # noqa
    print(f'using device: {device}')

    # source 可以是字符串、数组或张量等。
    results = model.predict(source=tryout_images, device=device,
                            conf=conf, iou=iou,
                            save=save,
                            # line_width=3
                            workers=8,  # workers 似乎不起作用
                            # half=True,   # half 似乎不起作用
                            )

    # print(f'{type(results)= }, {len(results)= }')

    # for i, result in enumerate(results):
        # print(f'{type(result)= }')
        # print(f'\n{result.names= }\n{result.keys= }')
        # print(f'{type(result.orig_img)= }, {result.orig_img.shape= }')
        # print(f'{result.orig_shape= }')
        # print(f'{type(result.masks)= }')
        # print(f'{result.path= }')
        # print(f'\n{type(result.boxes)= }')
        # print(f'{result.boxes.cls= }')
        # print(f'{result.boxes.conf= }')
        # print(f'{result.boxes.data= }')
        # print(f'result.boxes.xyxy= \n{result.boxes.xyxy}')
        # print(f'result.boxes.xyxyn= \n{result.boxes.xyxyn}')
        # print(f'result.boxes.xywh= \n{result.boxes.xywh}')
        # print(f'result.boxes.xywhn= \n{result.boxes.xywhn}')
        # print(f'{result.boxes= }')

        # print(f'{type(result.probs)= }')
        # if i == 0:
        #     break


if __name__ == '__main__':
    configurations = {'model_id': r'dx10',
                      # 'model_path': None,
                      # 'detect_data': 'coco128.yaml',
                      'batch': 4,
                      'nbs': 16,
                      'optimizer': 'Adam',
                      'lr0': 3.5e-4,  # 'lr0': sgd--3.87e-3,
                      # 'lrf': 0.01,  # 'lrf': 1
                      'epochs': 10}
    # train_on_solar_panel(configurations=configurations)  # 使用现有模型，继续训练

    model_path = r'~/work/cv/2023_05_24_infrared/pretrained_models/' \
                 r'dx08_Adam_lr3.50e-04_b4_e160_nbs16_val413_test470.pt'  # noqa
    tryout_images = r'~/work/cv/2023_05_24_infrared/dataset_solar_panel/images/test'
    get_prediction_result(model_path=model_path, tryout_images=tryout_images,
                          conf=0.5,
                          iou=0.7)  # 0.71 0.7。


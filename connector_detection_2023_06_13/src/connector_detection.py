#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于训练一个 YOLOv8 检测模型，使用光伏板红外图片进行检测。例如用于检测损坏的光伏板等。

版本号： 0.1
日期： 2023-05-30
作者： jun.liu@leapting.com
"""
import pathlib
import shutil
import time

import cv2 as cv
import torch
from ultralytics import YOLO

import utilities


@utilities.timer
def train_on_solar_panel(configurations=None,
                         rename_model=True, move_best_model=True, only_need_map50=True):
    """使用光伏板数据，训练 YOLOv8 的 object detection 模型。

    Arguments：
        configurations (dict): 一个字典，包含了部分需要修改的设置参数。
        only_need_map50 (bool): 布尔值，如果 only_need_map50 为 True，则最后只把 mAP50 指标记录到模型的名字中。
            这是因为对于插接件检测来说，位置精度要求不高，只要识别是否断开。
    """
    if configurations is None:
        configurations = {}
    model_path = configurations.get(
        'model_path', r'/media/disk_2/work/cv/2023_06_07_connector/pretrained_models/yolov8x.pt')
    model_path = pathlib.Path(model_path).expanduser().resolve()
    print(f'{model_path= }')

    target_model_path = r'/media/disk_2/work/cv/2023_06_07_connector/pretrained_models'
    target_model_path = pathlib.Path(target_model_path).expanduser().resolve()

    detect_data = configurations.get(
        'detect_data', r'/media/disk_2/work/cv/2023_06_07_connector/tryout/src/connector.yaml')
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

    imgsz = configurations.get('imgsz', 640)  # n114 x36
    hsv_h = configurations.get('hsv_h', 0.015)  # hue 为颜色，默认 0.015。
    hsv_s = configurations.get('hsv_s', 0.7)  # saturation 为饱和度，默认 0.7。

    hsv_v = configurations.get('hsv_v', 0.4)  # value 为亮度，默认 0.4。验证有晚上的场景时， 0.2, 0.9 无效。

    # nominal batch size，即训练数量达到 nbs，才会有一次梯度下降。
    nbs = configurations.get('nbs', batch)  # 验证分割时， nbs 64 的效果不错。

    degrees = configurations.get('degrees', 0)  # 7 的效果不错。默认为 0 。
    scale = configurations.get('scale', 0.5)  # 。默认为 0.5 。
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
    # fl_gamma = 2  #  验证 1.5, 2  无效。
    # 考虑加大 dfl loss gain，即加强 mask 部分的损失。因为目前 box 的 mAP 要好
    # 于 mask 的 mAP 。默认分类和 box 损失总和为 8，dfl 为 1.5，考虑加大为 7.5
    # box = 1.5  #  box 损失增益 gain。 验证 1.5 无效。
    # dfl = 1.0  #  验证 1 无效。
    epochs = configurations.get('epochs', 6)

    # close_mosaic 设置的是最后倒数若干个 epochs，关闭 mosaic。
    # 验证 50-200（epochs 为 200） 有效果。
    close_mosaic = configurations.get('close_mosaic', 0)  # 验证 epochs 300，配合 close 100 效果不错。
    # momentum = 0.9  #  验证 0.95， 0.9无效。
    # mask_ratio = 2  # 对 mask 进行下采样，验证 1, 2, 10 效果均不佳。
    # imgsz = 960  # 使用大的 size
    # weight_decay = 0.05
    # conf 和 iou 仅用于一个训练 epoch 结束之后，进行验证 val 时起作用。
    conf = configurations.get('conf', 0.5)
    iou = configurations.get('iou', 0.7)  # 老的 YOLOv8 版本，iou 为 0.7，现在新的为 0.6。

    experiment_name = f'{model_id}_{optimizer}_lr{lr0:.2e}_lrf{lrf:.2f}_conf{conf}_iou{iou}' \
                      f'_b{batch}_e{epochs}_nbs{nbs}_imgsz{imgsz}_hsv{hsv_h}_close_mos{close_mosaic}' \
                      f'_deg{degrees}_scale{scale}'
                      # f'_hsv'  # \

    model = YOLO(model_path)  # model_path
    # 这个 batch 是训练的 batch。验证的 batch 大小不受影响，似乎是默认值 16。

    model.train(
        data=str(detect_data), patience=0,  # 注意 data 参数必须输入字符串。
        name=experiment_name,
        optimizer=optimizer, lr0=lr0,
        lrf=lrf,
        # momentum=momentum,
        scale=scale,
        # mask_ratio=mask_ratio,
        epochs=epochs,
        close_mosaic=close_mosaic,
        batch=batch,
        imgsz=imgsz,
        degrees=degrees,
        nbs=nbs,
        # perspective=perspective,
        # shear=shear,
        # rect=rect,
        # copy_paste=copy_paste,
        # iou=iou,
        # cos_lr=cos_lr,
        # flipud=flipud,
        # translate=translate,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        # fl_gamma=fl_gamma,
        # box=box,
        # mixup=mixup,
        # dfl=dfl,
        # weight_decay=weight_decay,
        device=device,
        # 因为每个训练 epoch 之后，都要用验证集进行验证，所以可以用 conf, iou，使得
        # 保存 checkpoints 时，能按照指定 conf, iou 保存。
        conf=conf, iou=iou,
        agnostic_nms=True,  # 似乎要用 agnostic_nms 才能和下面验证 best model 保持一致。
        workers=8,  # 本机有 16 个处理器。
        # resume 似乎是指训练中途被打断的情况。例如设定 35 次，训练到 30 次被
        # 打断，然后再使用 resume，则会继续后面的 5 次。
        # resume=True,
        # pretrained=True,  # 似乎不起作用，可能和 YOLOv8 版本有关。
        cache=True,   # use cache to speed up 2.3 times faster
    )

    # 加载训练好的最佳模型，检查其 mAP50 指标。
    best_model_path = f'runs/detect/{experiment_name}/weights/best.pt'  # 模型的默认保存路径。
    best_model_path = pathlib.Path(best_model_path).expanduser().resolve()
    best_model = YOLO(best_model_path)

    # val 方法中不需要设置 imgsz，模型会自动设置（应该是使用训练时的 imgsz 参数）。
    # 设置小的 batch 以免爆显存，默认为 batch=16。
    metrics_validation = best_model.val(split='val', save=False,
                                        agnostic_nms=True, batch=batch,
                                        conf=conf, iou=iou)  # conf=0.5, iou=0.7
    metrics_test = best_model.val(split='test', save=False,
                                  agnostic_nms=True, batch=batch,
                                  conf=conf, iou=iou)
    if only_need_map50:
        map_val = round(metrics_validation.box.map50, 3)
        map_test = round(metrics_test.box.map50, 3)
    else:
        map_val = round(metrics_validation.box.map, 3)
        map_test = round(metrics_test.box.map, 3)
    if rename_model:
        # 在模型名字中加上 map 指标。
        new_name = best_model_path.parent / (
            f'{experiment_name}_val{map_val * 1000:03.0f}_test{map_test * 1000:03.0f}.pt')
        best_model_path = best_model_path.rename(new_name)
    if move_best_model:
        target_model_path = target_model_path / best_model_path.name
        shutil.move(best_model_path, target_model_path)
        print(f'Best model is saved as: {target_model_path}\n')
    else:
        print(f'Best model is saved as: {best_model_path}\n')


@utilities.timer
def get_prediction_result(model_path, tryout_images, conf=0.5, iou=0.7, imgsz=640,
                          agnostic_nms=True, save=True):
    """查看预测的输出结果。"""
    model_path = pathlib.Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    print(f'{model_path= }')
    model_name = model_path.stem.split('_')[0]  # 取名字第一个下划线的前面部分即可

    model = YOLO(model_path)  # 加载训练好的模型
    # print(f'{model.overrides = }\n')  # 是自定义的信息，包括 yaml 文件的路径，以及 imgsz 等。

    tryout_images = pathlib.Path(tryout_images).expanduser().resolve()
    if not tryout_images.exists():
        raise FileNotFoundError(f'Images not found: {tryout_images}')

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')  # noqa
    print(f'using device: {device}')

    # 时间格式 '230620_1016'，前半部分为日期，后半部分为小时和分钟
    time_stamp = time.strftime('%y%m%d_%H%M')
    # TODO: 后续考虑增加对 predict_name 存在时，即该文件夹存在时，抛出一个异常
    predict_name = f'predict_{model_name}_{time_stamp}_imgsz{imgsz}_conf{conf}_iou{iou}'  # 预测结果将保存在 predict_name 这个文件夹中
    # source 可以是字符串、数组或张量等。
    # 注意 model.predict 中可以设置 imgsz 参数，但是在 model.val 方法中设置 imgsz 无效。
    results = model.predict(source=tryout_images, device=device,
                            conf=conf, iou=iou,
                            save=save,
                            agnostic_nms=agnostic_nms,  # 每个接插件只能有一种状态，所以启用 agnostic_nms
                            # line_width=3
                            workers=8,  # workers 似乎不起作用
                            name=predict_name,  # 记录下名字，以便查看结果
                            imgsz=imgsz,  # 记录下名字，以便查看结果
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


def test_best_model(best_model_path, conf, iou, agnostic_nms, imgsz, batch_size=1, half=False):
    """test and val on best model.
    """
    best_model_path = pathlib.Path(best_model_path).expanduser().resolve()
    if not best_model_path.exists():
        raise FileNotFoundError(f'Model not found: {best_model_path}')
    print(f'{best_model_path= }')
    best_model = YOLO(best_model_path, task='detect')  # 须在创建模型时设置 task。
    if best_model_path.suffix in ['.engine', '.onnx']:
        # 使用 engine 推理时，必须设置 imgsz，data 和 task，且 data 必须为字符串。
        # half 参数对 TensorRT 推理没有影响，所以在 val 方法中直接去掉。
        detect_data = r'/media/disk_2/work/cv/2023_06_07_connector/tryout/src/connector.yaml'
        metrics_validation = best_model.val(split='val', save=False,
                                            agnostic_nms=agnostic_nms, batch=batch_size,
                                            conf=conf, iou=iou, imgsz=imgsz,
                                             data=detect_data)
        metrics_test = best_model.val(split='test', save=False,
                                      agnostic_nms=agnostic_nms, batch=batch_size,
                                      conf=conf, iou=iou, imgsz=imgsz,
                                      data=detect_data)
    else:
        # YOLOv8 的 pt 模型带有 imgsz 等参数，无须在 val 方法中设置（应该是使用训练时的 imgsz 参数）。
        metrics_validation = best_model.val(split='val', save=False,
                                            agnostic_nms=agnostic_nms, batch=batch_size,
                                            conf=conf, iou=iou,
                                            half=half)
        metrics_test = best_model.val(split='test', save=False,
                                      agnostic_nms=agnostic_nms, batch=batch_size,
                                      conf=conf, iou=iou,
                                      half=half)

    map50_val = round(metrics_validation.box.map50, 3)
    map50_test = round(metrics_test.box.map50, 3)
    print(f'{map50_val= },  {map50_test= }')


def export_yolo(model_path, format='engine', half=False, workspace=8):
    """把 YOLOv8 模型从 .pt 格式转换为 TensorRT 的 engine 格式。
    """
    model_path = pathlib.Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    print(f'{model_path= }')
    model = YOLO(model_path)

    # 转换为 TensorRT 的 engine 时，需要设置 device=0。设置 imgsz 参数没有任何作用。
    model.export(format=format, device=0,
                 half=half,
                 int8=True, workspace=workspace)

    # 把导出的 engine 重新命名。
    exported_engine_name = model_path.stem + f'.{format}'
    exported_engine = model_path.parent / exported_engine_name

    data_type = 'FP16' if half else 'FP32'
    # data_type = 'int8'  直接使用 int8 无效，可能要用 trtexec
    new_engine_name = model_path.stem + f'_{data_type}.{format}'
    renamed_engine = model_path.parent / new_engine_name
    exported_engine.rename(renamed_engine)

    exported_onnx_name = model_path.stem + f'.onnx'
    exported_onnx = model_path.parent / exported_onnx_name
    if exported_onnx.exists():  # ONNX 存在时，也将其改名。
        new_onnx_name = model_path.stem + f'_{data_type}.onnx'
        renamed_onnx = model_path.parent / new_onnx_name
        exported_onnx.rename(renamed_onnx)
    print(f'Done! Model is exported as {renamed_engine}')


if __name__ == '__main__':
    configurations = {'model_id': r'exp70',  # x22-connector exp61
                      # 'model_path': None,
                      # 'detect_data': 'coco128.yaml',
                      'conf': 0.5,
                      'iou': 0.7,
                      'batch': 2,  # 4
                      'imgsz': 1088,  # 480 1280  应该是 32 的倍数.
                      'hsv_h': 0.8,  # 0.2
                      'hsv_s': 0.8,  # 0.2
                      'hsv_v': 0.8,
                      'close_mosaic': 1,
                      'degrees': 7,
                      'scale': 0.5,
                      # 'nbs': 16,
                      'optimizer': 'Adam',
                      'lr0': 1.08e-5,  # 1.08e-5  8e-5
                      'lrf': 0.01,  # 'lrf': 1
                      'epochs': 25}  # 500
    train_on_solar_panel(configurations=configurations)  # 训练模型

    # 用训练好的模型对测试集图片进行预测 dx007_Adam_lr1.08e-05_lrf0.01_conf0.5_iou0.7_b4_e8000_nbs4_val914_test890
    model_path = r'/media/disk_2/work/cv/2023_06_07_connector/pretrained_models/' \
                 r'x22-connector_Adam_lr1.08e-05_lrf0.01_conf0.5_iou0.7_b2_e500_nbs2_imgsz1088_hsv0.8_close_mosaic1_deg0_val945_test924.pt'  # noqa
                 # r'dx014_Adam_lr1.08e-05_lrf0.01_conf0.5_iou0.7_b2_e550_nbs2_imgsz1088_hsv-v0.8_hsv-h0.2_val971_test962_FP16.engine'  # noqa
    # export_yolo(model_path, format='engine', half=True)
    # test_best_model(best_model_path=model_path, batch_size=2,
    #                 half=False, imgsz=1088,  # 1088
    #                 conf=0.5, iou=0.7, agnostic_nms=True)  # test and validate on the best model

    tryout_images = r'/media/disk_2/work/cv/2023_06_07_connector/dataset_connector/images/' \
                    r'test'  # validation test
    # tryout_images = r'/media/disk_2/work/cv/2023_06_07_connector/videos/captured_frames_高清'
    tryout_images = r'/media/disk_2/work/cv/2023_06_07_connector/documents/2023-07-05 测光方式对比'
    # tryout_images = r'/media/disk_2/work/cv/2023_06_07_connector/dataset_connector/' \
    #                 r'original_data/2023-06-18 湖州图片/images'
    # # # # TODO: 给推理程序 get_prediction_result 加上解码名字的部分，确保 conf，iou 等参数设置正确。
    # get_prediction_result(model_path=model_path, tryout_images=tryout_images,
    #                       imgsz=1088,  # default imgsz=640
    #                       conf=0.5,  # default 0.25 predict, 0.001 val
    #                       iou=0.7,
    #                       agnostic_nms=True)  # 0.71 0.7。


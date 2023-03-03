#!/usr/bin/env python3
# -*- coding utf-8 -*-
"""本文件用于 YOLOv8 的分割演示。
如需支持，可联系 jun.liu@leapting.com
"""
import time

from ultralytics import YOLO


def train_on_solar_panel():

    # Load an inner_model
    # inner_model = YOLO("yolov8n.yaml")  # build a new inner_model from scratch
    # 预训练的分割模型，要加上 -seg
    # model_segmentation = r'../pretrained_models/yolov8n-seg.pt'
    model_segmentation = r'../pretrained_models/yolov8x-seg.pt'
    model = YOLO(model_segmentation)

    # 对于检测模型的数据，使用 coco128.yaml。对于分割的数据，也要加 -seg。
    # results = inner_model.train(data="coco128.yaml", epochs=3)  # train the inner_model
    segment_data = r'../solar_panel_data/solar_panel_full_data.yaml'
    # 因为显存不够大，batch_size 应该用 8 或者更小。 lrf=1e-3, plots=True, show=True
    # 如果使用 Adam 优化器，学习率至少要小于 1e-3 。
    # 这个 batch 是训练的 batch。验证的 batch 大小不受影响，似乎是默认值 16。
    results = model.train(data=segment_data, patience=0,
                          batch=8, epochs=50, optimizer='Adam', lr0=1e-3,
                          )

    # 验证训练好的模型，用的是 validation 文件夹的数据，并且 batch 大小也是
    # 上面训练时设置好的，无须再设置。但是没有 inner_model.test() 方法。
    # inner_model.val()  # 训练结束后会自动调用 val。
    # results = inner_model.train(data=r"tryout.yaml", batch=4, epochs=3)
    # results = inner_model.val()  # evaluate inner_model performance on the validation set

    # 有 2 种方法进行 predict，如下为第 1 种。
    # results = inner_model("https://ultralytics.com/images/bus.jpg")
    # 第 2 种方法进行 predict。
    # results = inner_model.predict(source="../box.mp4", show=True, conf=0.5, line_thickness=3)

    # success = inner_model.export(format="onnx")  # export the inner_model to ONNX format


def get_prediction_result():
    """查看预测的输出结果张量。
    """
    trained_model = r'pretrained_models/yolov8x-seg.pt'
    # trained_model = r'solar_all_data_xseg_100.pt'
    model = YOLO(trained_model)  # 加载训练好的模型

    # tryout_images = r'../solar_panel_data/tryout_images'
    tryout_images = r'bus.jpg'
    results = model.predict(source=tryout_images,
                            # save=True,
                            conf=0.8, line_thickness=3)

    # inner_model.export(format='onnx')  # 根据需要导出中间格式。
    print(f'prediction results: type(results): {type(results)}, '
          f'len(results): {len(results)}')
    print(f'type(results[0]): {type(results[0])}')
    print(f'results[0].orig_img: {type(results[0].orig_img)}, {results[0].orig_img.shape}')
    # print(f'results[0].orig_shape: {type(results[0].orig_shape)}, {results[0].orig_shape}')
    # print(f'results[0].path: {results[0].path}')
    print(f'results[0]._keys: {results[0]._keys}')
    # print(f'results[0].names: {type(results[0].names)}， {results[0].names}')
    if results[0].boxes.shape[0] > 0:
        print(f'results[0].probs: {type(results[0].probs)}, {results[0].probs}')
        print(f'results[0].boxes: {type(results[0].boxes)}, {results[0].boxes.shape}')

        print(f'results[0].boxes: \n{results[0].boxes}')
        print(f'results[0].boxes.cls: {results[0].boxes.cls}')
        print(f'results[0].boxes.conf: {results[0].boxes.conf}')
        print(f'results[0].boxes.xywh: \n{results[0].boxes.xywh}')
    if results[0].masks is not None:
        print(f'results[0].orig_shape: {type(results[0].orig_shape)}, {results[0].orig_shape}')

        print(f'results[0].masks: {type(results[0].masks)}, {results[0].masks.shape}')
        print(f'results[0].masks.segments: {type(results[0].masks.segments)}, '
              f'{len(results[0].masks.segments)}')
        print(f'results[0].masks.segments[0]: '
              f'{type(results[0].masks.segments[0])}, {results[0].masks.segments[0].shape}')
        # print(f'results[0].masks: ', results[0].masks[0, 200:230, 300:320])


if __name__ == '__main__':
    train_on_solar_panel()  # 训练模型

    # 下面部分为预测。
    # tic = time.perf_counter()
    #
    # get_prediction_result()
    #
    # toc = time.perf_counter()
    # duration = (toc - tic) / 1000
    # print(f'duration: {duration} seconds')




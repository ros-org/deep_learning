#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于训练一个 RegNet 模型，然后对光伏板进行分类预测。主要区分光伏板上有异物和
没有异物两种情况。

版本号： 0.4
日期： 2023-05-24
作者： jun.liu@leapting.com
"""
import concurrent
import os
import pathlib
import shutil
import time
import unicodedata

import cv2 as cv
import numpy as np
import torch
import torchview
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import utilities

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if device != "cuda":
    raise ValueError('Not using GPU! Please check the CUDA status.')

dataset_size = '4k'  # 数据集总量为 4k，后面加到模型的名字中


class ForeignMatterClassifier(torch.nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1  # noqa
        regnet_y_16gf = torchvision.models.regnet_y_16gf(weights=weights)

        self.regnet_y_16gf_base = create_feature_extractor(
            regnet_y_16gf, return_nodes={'flatten': 'flatten'})

        self.solar_panel_classifier = torch.nn.Linear(3024, classes)

    def forward(self, x):
        # regnet_y_16gf_base 返回的是一个字典，并且 key 为 flatten。
        # 使用 x['flatten'] 的张量，进行前向传播，这种训练方式，
        # 是否会导致只有 solar_panel_classifier 的参数得到训练？待查。
        x = self.regnet_y_16gf_base(x)['flatten']
        x = self.solar_panel_classifier(x)

        return x


def get_and_extract_model():
    """Set up the model.

    Arguments:
        x: (1, 3, 224, 224)
    """

    weights = torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1
    regnet_y_16gf = torchvision.models.regnet_y_16gf(weights=weights)
    print(f'regnet_y_16gf: \n{regnet_y_16gf}')

    # To assist you in designing the feature extractor you may want to print out
    # the available nodes for resnet50.
    train_nodes, eval_nodes = get_graph_node_names(regnet_y_16gf)
    print(f'train_nodes: \n{train_nodes}')
    torchview.draw_graph(regnet_y_16gf, input_size=(1, 3, 224, 224),
                         save_graph=True, expand_nested=True,
                         filename='regnet_y_16gf_original',
                         device='meta')
    return_nodes = {'flatten': 'regnet_base'}
    regnet_extracted = create_feature_extractor(regnet_y_16gf,
                                                return_nodes=return_nodes)
    torchview.draw_graph(regnet_extracted, input_size=(1, 3, 224, 224),
                         save_graph=True, expand_nested=True,
                         filename='regnet_extracted',
                         device='meta')
    train_nodes_extracted, _ = get_graph_node_names(regnet_extracted)

    print(f'{unicodedata.lookup("bouquet")}' * 20)
    print(f'train_nodes_extracted: \n{train_nodes_extracted}')
    print(f'{unicodedata.lookup("bouquet")}' * 20)

    return regnet_extracted


def check_model():
    classifier_solar_panel = ForeignMatterClassifier(classes=2)
    print(f'classifier:\n{classifier_solar_panel}')
    train_nodes, eval_nodes = get_graph_node_names(classifier_solar_panel)
    print(f'train_nodes: \n{train_nodes}')

    torchview.draw_graph(classifier_solar_panel, input_size=(1, 3, 224, 224),
                         save_graph=True, expand_nested=True,
                         filename='classifier_solar_panel',
                         device='meta')

    get_and_extract_model()  # 查看 regnet_y_16gf 原始模型。


def train(dataloader, model, loss_fn, optimizer, scheduler=None):
    """train model."""
    # 创建进度条 tqdm 之后，只能显示一次。因此每次使用前都要创建一遍。
    dataloader = tqdm(dataloader, total=len(dataloader), ncols=80)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()  # 反向传播，计算梯度。
        optimizer.step()  # 更新参数。

    if scheduler is not None:
        scheduler.step()  # 改变学习率

    # loss = loss.item()
    # print(f"Training loss: {loss:>7f}")


def test(dataloader, model, loss_fn, dataset_type='validation',
         save_incorrect_results=False, results_dir=None, classes=None):
    images_quantity = len(dataloader.dataset)  # 注意用 dataset 属性，可以获得原始的 dataset
    num_batches = len(dataloader)
    tqdm_dataloader = tqdm(dataloader, total=num_batches, ncols=80)

    model.eval()
    test_loss, correct = 0, 0

    batch_count = 0  # 记录批次的数量，便于后面计算图片的索引值。
    tally_errors = 0  # 记录预测错误的图片总数。
    images_list = dataloader.dataset.imgs  # dataset.imgs 返回的是图片列表，包含了图片路径和标签
    with torch.no_grad():
        for X, y in tqdm_dataloader:
            X, y = X.to(device), y.to(device)
            # pred 的形状为 [batch_size, 2]
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # errors 是一个布尔张量，形状为 (batch_size,)，预测错误的图片在其中被记录为 True
            errors = (pred.argmax(1) != y)
            tally_errors += errors.sum()  # noqa 累计每个批次中错误的数量

            if save_incorrect_results:
                # index_error 张量形状为 [n, 1]，记录的是当前批次中，预测错误的图片索引，表示有 n 个图片预测错误
                index_errors = torch.nonzero(errors)  # noqa
                # 下面把每一个预测错误的图片保存到文件夹中。
                for index_error in index_errors:
                    image_index = batch_count * dataloader.batch_size + index_error
                    one_img_path = images_list[image_index][0]  # 第 0 位是图片路径，第 1 位是标签
                    # 因为前面创建为 dataset 时已经解析为绝对路径，所以这里无须再使用 resolve
                    one_img_path = pathlib.Path(one_img_path)
                    # one_prediction 张量形状为 [1, 2]
                    one_prediction = pred[index_error]
                    save_image(img_path=one_img_path, results_dir=results_dir,
                               prediction=one_prediction, classes=classes)

            batch_count += 1  # 注意应该在每个 batch 末尾再加一，否则 image_index 会超出列表的范围

    print(f'Total incorrect predictions: {tally_errors} . Total images: {images_quantity}')

    test_loss /= num_batches
    current_accuracy = correct / images_quantity
    print(f"{dataset_type.title()} Error: \n Accuracy: {current_accuracy:>0.1%}, Avg loss: {test_loss:>8f} \n")

    return current_accuracy, test_loss


def save_checkpoints(model, highest_accuracy_path, current_accuracy, highest_accuracy):
    if current_accuracy > highest_accuracy:

        torch.save(model, highest_accuracy_path)
        highest_accuracy = current_accuracy
        print(f'New record! Validation accuracy = {highest_accuracy:.1%}. Model is saved as: {highest_accuracy_path}\n')
    return highest_accuracy


def save_image(img_path, results_dir, prediction, classes):
    """把 RegNet 的预测概率值写到对应图片中，并保存到路径 :attr:`results_dir` 中。

    Arguments:
        img_path (pathlib.Path): 一个 Path 对象，指向一张图片。可以是相对路径或绝对路径。
        results_dir (pathlib.Path): 一个 Path 对象，指向一个文件夹。最终图片将被保存到这个文件夹中。
        prediction (torch.Tensor): 一个张量，是 RegNet 的预测结果，形状为 [1, 2]。
        classes (list[str, str]): 一个列表，包含了模型需要区分的类别名称。目前为 2 类，即只区分有异物和无异物。
    """
    # 将预测结果转换为概率值。 probability 形状为 [1, 2]
    probability = torch.sigmoid(prediction)
    class_idx = torch.argmax(probability)
    probability = f'{probability[0, class_idx]:.1%}'  # 从张量中得到最大的概率值。

    # print(f'{each_image.stem}, {probability.shape}, {probability}')
    class_name = classes[class_idx]

    img = cv.imread(str(img_path))  # noqa， OpenCV 需要输入字符串格式
    text = f'{class_name}, {probability}'
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

    result_img = results_dir / f'{img_path.stem}_pred.jpg'
    if result_img.exists():
        os.remove(result_img)  # 如果已经存在结果，则删除
    cv.imwrite(str(result_img), img)  # noqa， OpenCV 需要输入字符串格式


def load_image_worker(one_image_path):
    """使用单个线程，加载路径 :attr:`one_image_path` 中的图片。

    Arguments:
        one_image_path (pathlib.Path): 一个 Path 对象，指向一张图片。

    Returns:
        pil_image (torch.Tensor)： 一个类型为 uint8 图片张量，形状为 (depth, height, width)。
        one_image_path (pathlib.Path)： 是输入的 Path 对象。
    """
    if one_image_path.is_file():
        # read_image 要求输入的为字符串类型，因此用 str 转换一下。
        # pil_image 是一个 uint8 图片张量，形状为 (depth, height, width)。
        pil_image = torchvision.io.read_image(str(one_image_path))
        return pil_image, one_image_path


@utilities.timer
def prediction(model_path, image_folder, task='prediction', save_incorrect_results=False,
               batch_size=1, overwrite_results=True, imgsz=640):
    """使用 RegNet 模型，对文件夹 :image_folder:`image_folder` 内图片进行预测。也可以使用测试集，计算模型的准确度。

    Arguments:
        model_path (str): 一个字符串，指向一个 RegNet 模型，模型后缀为 .pt 或 .pth。
        image_folder (str): 一个字符串，指向一个文件夹。如果是预测图片，则文件夹中存放的是图片。
            如果是用来计算准确度的测试集，则该文件夹内应该有一个名为 test 的文件夹，test 文件夹内有 2 个子文件夹，分别
            叫做 OK 和 foreign_matter，OK 文件夹放没有异物的图片，foreign_matter 中放有异物的图片。
        task (str): 一个字符串，是 'prediction' 或 'test_accuracy'，表示 2 种任务。
        save_incorrect_results (bool): 布尔值。如果为 True，则在 test_accuracy 任务中，会把预测错误的图片和预测
            结果一起保存下来。
        batch_size (int): 一个整数，是批量的大小。
        overwrite_results (bool): 布尔值。如果为 True，则会在训练过程中，把验证集准确度最高的模型保存下来。
        imgsz (int): 一个整数，是图片缩放后的大小，即图片会被缩放到这个大小的高度和宽度，然后进行预测。这个参数
            必须和训练时的保持一致，否则预测会出错。模型 .pt 文件的名字会包含 imgsz 信息，可以根据模型名字设置其大小。
    """
    if task not in ['prediction', 'test_accuracy']:
        raise ValueError(f'Only "prediction, test_accuracy" are valid now for task, but {task=} now.')

    model_path = pathlib.Path(model_path).expanduser().resolve()
    model = torch.load(model_path)  # TODO: 后续考虑改为加载权重的模式。
    print(f'Using model: {model_path}')
    image_folder = pathlib.Path(image_folder).expanduser().resolve()
    print(f'Checking images in: {image_folder}')

    results_dir = None
    if save_incorrect_results:
        # 时间格式 '20230512_1016'，前半部分为日期，后半部分为小时和分钟
        time_suffix = time.strftime('%Y%m%d_%H%M')
        results_folder = r'~/work/cv/2023_05_04_regnet/tryout/prediction_results'
        results_folder = pathlib.Path(results_folder).expanduser().resolve()
        results_dir = results_folder / f'results_{time_suffix}'
        if results_dir.exists() and overwrite_results:
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True)
        print(f'Prediction results will be save to: \n{results_dir}')

    # get_class_id_only 时，应该是总数据集的文件夹，这样才能正确找到各个类别。
    full_data_root = '~/work/cv/2023_05_04_regnet/classification_data'
    classes = get_data(data_root=full_data_root, get_class_id_only=True)

    if task == 'test_accuracy':
        # 预测时不要设置 dataset_type
        dataset = get_dataset(image_folder, dataset_type='test', imgsz=imgsz)
        # Create data loaders.
        dataloader = DataLoader(dataset, batch_size=batch_size)

        loss_fn = torch.nn.CrossEntropyLoss()
        test(dataloader, model, loss_fn, dataset_type='test', save_incorrect_results=save_incorrect_results,
             results_dir=results_dir, classes=classes)
    elif task == 'prediction':
        model.eval()
        images_quantity = 0
        tally_load_image = 0
        tally_preprocess = 0
        tally_inference = 0
        tally_postprocess = 0
        with torch.no_grad():
            # 如果要用 tqdm，就要知道文件的数量。但是因为 Path.iterdir() 没有 __len__ 方法，
            # 所以要用 os.listdir() 得到文件数量。
            folder_size = len(os.listdir(image_folder))
            # images_iterator = tqdm(image_folder.iterdir(), total=folder_size, ncols=80)

            # ============ 使用多线程并发，读取图片 ===================================

            preprocess_tic = time.perf_counter()  # 在加载图片之前进行计时。

            with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:  # noqa
                results = executor.map(load_image_worker, image_folder.iterdir())

            images_iterator = tqdm(results, total=folder_size, ncols=80)

            for i, result in enumerate(images_iterator):
                if result is not None:  # 如果遍历文件夹时遇到子文件夹，result 会是一个 None
                    pil_image, each_image = result
                    # if pil_image is not None:
                    # pil_image = each_image

                    #         print(f'{i=}, \t{each_image.shape=}')
                    #
                    # for each_image in images_iterator:
                    #     if each_image.is_file():

                    # tic = time.perf_counter()
                    images_quantity += 1
                    # read_image 要求输入的为字符串类型，因此用 str 转换一下。
                    # pil_image 是一个 uint8 图片张量，形状为 (depth, height, width)。
                    # pil_image = torchvision.io.read_image(str(each_image))

                    t_load_image = time.perf_counter()  # 加载图片时间点

                    # 1. 建议先把图片移到 GPU 上，再进行前处理。以下部分前处理约为 3ms。
                    pil_image = pil_image.to(device)

                    # pil_image 形状为 (depth, height, width)，数据类型为 float32，
                    # 并且数值被转换为 [0, 1]。
                    pil_image = torchvision.transforms.ConvertImageDtype(torch.float32)(pil_image)
                    # pil_image 的大小必须和模型要求的输入保持一致。
                    # 设置 antialias=True 是为了避免错误信息的提示。
                    pil_image = torchvision.transforms.Resize(
                        (imgsz, imgsz), antialias=True)(pil_image)
                    pil_image = pil_image[None, ...]  # 预测时，需要输入 4D 张量

                    # 2. 如果前处理之后再把图片移到 GPU 上，则前处理的时间约为 26ms。
                    # pil_image = pil_image.to(device)
                    t_preprocess = time.perf_counter()  # 前处理时间点

                    pred = model(pil_image)  # 进行预测
                    t_inference = time.perf_counter()

                    save_image(img_path=each_image, results_dir=results_dir,
                               prediction=pred, classes=classes)
                    t_postprocess = time.perf_counter()

                    # 因为是用多线程并发加载图片，所以要改变加载时间的计算方式，即变为从一个循环结束为起点，
                    # 加载到图片之后为终点，两者之差是图片的加载时间。
                    if i == 0:
                        duration_load_image = t_load_image - preprocess_tic
                    else:
                        duration_load_image = t_load_image - t_circle_end

                    duration_preprocess = t_preprocess - t_load_image
                    duration_inference = t_inference - t_preprocess
                    duration_postprocess = t_postprocess - t_inference

                    tally_load_image += duration_load_image
                    tally_preprocess += duration_preprocess
                    tally_inference += duration_inference
                    tally_postprocess += duration_postprocess

                    # 在使用多线程并发时，对于加载图片的时间，需要改变计算方式，所以下面要
                    # 记录这个时间点 t_circle_end
                    t_circle_end = time.perf_counter()

        if images_quantity > 0:  # 文件夹内有图片时才计算处理时间。
            average_load_image = 1000 * tally_load_image / images_quantity
            average_preprocess = 1000 * tally_preprocess / images_quantity
            average_inference = 1000 * tally_inference / images_quantity
            average_postprocess = 1000 * tally_postprocess / images_quantity
            print(f'Total images: {images_quantity}')
            print(f'Average duration: \n'
                  f'load image: {average_load_image:.0f} ms, '
                  f'preprocess: {average_preprocess:.0f} ms, '
                  f'inference: {average_inference:.0f} ms, '
                  f'save image: {average_postprocess:.0f} ms')
    print('Done!')


def show_dataset(dataset, quantity=3):
    print(f'train_dataloader: {type(dataset)}, {len(dataset)}')
    for i, data in enumerate(dataset):
        print(f'each in train_dataloader: {type(data)} , {len(data)}')  #
        x, y = data
        print(f'sample {i}, x: {type(x)} , {x.shape}')  #
        print(f'label {i},  y: {type(y)}, {y}')
        if quantity is not None:
            if i == quantity:
                break


def get_data(data_root=None, batch_size=8, get_class_id_only=False, imgsz=None):
    if data_root is None:
        data_root = pathlib.Path(r'dataset_demo').expanduser().resolve()
    data_root = pathlib.Path(data_root).expanduser().resolve()
    # root = r'/home/leapting/work/cv/2023_05_06_datasets/kaggle_train_test'

    test_data = get_dataset(data_root, dataset_type='test', imgsz=imgsz)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print(f'Classes: {test_data.classes}')

    if get_class_id_only:
        output = test_data.classes
    else:
        training_data = get_dataset(data_root, dataset_type='train', imgsz=imgsz)
        validation_data = get_dataset(data_root, dataset_type='validation', imgsz=imgsz)
        # Create data loaders.
        # 因为自定义的光伏板数据集是固定放在 2 个文件夹，加载时必须打乱顺序，避免
        # 前一半全是 A 类别，后一半全是 B 类别的情况。如果不打乱，会使得前面几轮
        # 训练效果较差。
        train_dataloader = DataLoader(training_data,
                                      batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(validation_data,
                                           batch_size=batch_size, shuffle=True)

        output = train_dataloader, validation_dataloader, test_dataloader

    # show_dataset(train_dataloader, 3)  # 查看数据集内部。

    return output


def get_dataset(root: pathlib.Path, dataset_type=None, imgsz=None):
    # root = pathlib.Path(root)
    if dataset_type is not None:
        dataset_path = root / dataset_type  # dataset_type 是 'train'，'test' 等
    else:
        dataset_path = root
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                        std=[0.229, 0.224, 0.225])
    if dataset_type == 'train':
        # 训练数据需要做数据增加，因此使用 ColorJitter 等。
        image_transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),  # 图片改为 imgsz x imgsz 大小
            transforms.ToTensor(),
            # ColorJitter 是否需要在 ToTensor 之前？
            transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                   saturation=0.5, hue=0.4),
            transforms.RandomHorizontalFlip(0.5),  # 水平和竖直翻转。
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # 视角变换的数据增强
        ])
    else:
        image_transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ])

    dataset = torchvision.datasets.ImageFolder(
        root=dataset_path, transform=image_transform)

    # print(f'dataset: {type(dataset)}')
    # print(f'dataset.classes: {dataset.classes}')
    # print(f'dataset.class_to_idx: {dataset.class_to_idx}')
    # print(f'dataset.imgs: {dataset.imgs}')
    # show_dataset(dataset)

    return dataset


@utilities.timer
def main(data_root, lr=1.1e-2, batch_size=8, epochs=5, classes=2,
         imgsz=640,
         optimizer_name='SGD', momentum=0.9,
         step=10,
         checkpoint_path=None, save_model=True):
    """创建一个 RegNet 模型，并且对其进行训练。

    Arguments:
        data_root (str): 一个字符串，指向一个文件夹，其中放着训练集，验证集和测试集 3 个文件夹。
        lr (float): 一个浮点数，是模型的学习率。
        batch_size (int): 一个整数，是批量的大小。
        epochs (int): 一个整数，是训练的迭代次数。
        classes (int): 一个整数，是模型需要区分的类别数量。目前为 2 类，即只区分有异物和无异物。
        imgsz (int): 一个整数，是图片缩放后的大小，即图片会被缩放到这个大小的高度和宽度，然后进行训练。
        optimizer_name (str): 一个字符串，是 'SGD' 或 'Adam'，表示使用的优化器名字。
        checkpoint_path (str): 一个字符串，指向一个文件夹，训练好的最佳模型将存放在此路径中。
        save_model (bool): 布尔值。如果为 True，则会在训练过程中，把验证集准确度最高的模型保存下来。
    """
    # 保存到 checkpoints
    if checkpoint_path is None:
        checkpoint_path = pathlib.Path(__file__).parent / 'checkpoints'
    else:
        checkpoint_path = pathlib.Path(checkpoint_path).expanduser().resolve() / 'checkpoints'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir()
    print(f'Model will be saved to: {checkpoint_path}')
    if optimizer_name not in ['SGD', 'Adam']:
        raise ValueError(f'Only "SGD, Adam" are configured now for optimizer, but {optimizer_name=}')

    # 时间格式 '20230512_1016'，前半部分为日期，后半部分为小时和分钟
    time_suffix = time.strftime('%Y%m%d_%H%M')
    experiment_name = f'{time_suffix}_{dataset_size}_{optimizer_name}_lr{lr:.2e}_b{batch_size}' \
                      f'_e{epochs}_step{step}_hue0.4_imgsz{imgsz}'  # \_m{momentum}_nesterov
    # f'degrees{degrees}_nbs{nbs}_close_mosaic{close_mosaic}_'
    model_name = f'{experiment_name}.pt'
    highest_accuracy_path = checkpoint_path / model_name

    experiment_dir = checkpoint_path / experiment_name

    # SummaryWriter 会写入 checkpoints 文件夹下。
    writer = SummaryWriter(log_dir=str(experiment_dir))

    highest_accuracy = 0  # 记录最高准确度，据此保存模型。

    # {momentum=}
    print(f'{lr=:.2e}, {batch_size=}, {optimizer_name=}, {imgsz=}, \n{epochs=}, {step=}')

    # input_shape = (3, 32, 32)
    # size = np.prod(input_shape)
    model = ForeignMatterClassifier(classes).to(device)  # 注意把模型移到 device 上
    # print(model)

    # Create data loaders.
    train_dataloader, validation_dataloader, test_dataloader = get_data(
        data_root, batch_size, imgsz=imgsz)

    # dubug 时可以查看样本和标签的形状。
    # for X, y in test_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break

    loss_fn = torch.nn.CrossEntropyLoss()
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # , momentum=momentum, nesterov=True
        scheduler = lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step, gamma=0.5)

    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, scheduler)  # noqa
        current_accuracy, loss = test(validation_dataloader, model, loss_fn)

        # 记录准确度指标
        writer.add_scalar('Accuracy/validation', current_accuracy, t + 1)
        writer.add_scalar('Loss/validation', loss, t + 1)
        if save_model:
            highest_accuracy = save_checkpoints(
                model=model, highest_accuracy_path=highest_accuracy_path,
                current_accuracy=current_accuracy,
                highest_accuracy=highest_accuracy)

    best_model = torch.load(highest_accuracy_path)
    validation_accuracy, _ = test(validation_dataloader, best_model, loss_fn, dataset_type='test')
    test_accuracy, _ = test(test_dataloader, best_model, loss_fn, dataset_type='test')
    print(f"Best accuracy: \tvalidation: {validation_accuracy:.1%}, test: {test_accuracy:.1%}")
    if save_model:
        # 加上测试集准确度之后再保存一次模型。
        add_accuracy = highest_accuracy_path.parent / (
                highest_accuracy_path.stem +
                f'_val{validation_accuracy * 1000:.0f}_test{test_accuracy * 1000:.0f}.pt')
        highest_accuracy_path.rename(add_accuracy)
        print(f'Best model is saved as: {add_accuracy}\n')

    writer.flush()  # 把数据写入硬盘。
    writer.close()  # 关闭。
    print("Done!")


if __name__ == '__main__':
    # 训练模型
    data_root = '~/work/cv/2023_05_04_regnet/classification_data'
    main(data_root=data_root,
         lr=5e-3, batch_size=2,  # 1.1e-2
         imgsz=640,
         epochs=27, step=8)

    # 测试准确度
    #  RegNet_20230523_1058_testacc_989 20230523_1341_3k_SGD_lr1.10e-02_b8_e15_step10_testacc_985
    #  20230523_1553_3k_SGD_lr5.00e-03_b8_e12_step10_jitter_testacc_989
    # # image_folder = r'~/work/cv/2023_05_04_regnet/classification_data'

    # 20230523_1958_4k_SGD_lr5.00e-03_b8_e25_step10_jitter_hue0.4_testacc_995
    # model_path = r'checkpoints/' \
    #              r'20230523_2312_4k_SGD_lr1.10e-02_b8_e125_step30_hue0.4_imgsz300_val987_test996.pt'
    # image_folder = r'~/work/cv/2023_05_04_regnet/tryout/random_test'
    # prediction(model_path=model_path, image_folder=image_folder,
    #            task='test_accuracy', imgsz=300,  # 注意 5//24 之后的模型可能用 640 大小图片
    #            save_incorrect_results=True,
    #            batch_size=8)

    # 对文件夹内图片进行预测
    # image_folder = r'prediction_images'
    # prediction(model_path=model_path, image_folder=image_folder,
    #            batch_size=8)

    # load_image_multithreading(image_folder)




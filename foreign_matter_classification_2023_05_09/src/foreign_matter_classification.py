#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于训练一个 RegNet 模型，对光伏板进行分类。主要区分光伏板上有异物和
没有异物两种情况。

版本号： 0.2
日期： 2023-05-09
作者： jun.liu@leapting.com
"""
import os
import pathlib
import shutil
import time
import unicodedata

import cv2 as cv
import graphviz
import numpy as np
import torch
import torchview
from torchvision import transforms
from torch.utils.data import DataLoader

import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import utilities

graphviz.set_jupyter_format('png')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class ForeignMatterClassifier(torch.nn.Module):
    def __init__(self, classes):
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


def train(dataloader, model, loss_fn, optimizer):
    """train model."""

    size = len(dataloader.dataset)
    tqdm_dataloader = tqdm(dataloader, total=len(dataloader), ncols=80)
    model.train()
    for batch, (X, y) in enumerate(tqdm_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()  # 反向传播，计算梯度。
        optimizer.step()  # 更新参数。

        # if batch % 5 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    loss = loss.item()
    # print(f"Training loss: {loss:>7f}")


def test(dataloader, model, loss_fn, data_type='validation',
         save_results=False, results_dir=None, classes=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    tqdm_dataloader = tqdm(dataloader, total=len(dataloader), ncols=80)

    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # if save_results:
            #     save_image(img_path=each_image, results_dir=results_dir,
            #                prediction=pred, classes=classes)
    test_loss /= num_batches
    current_accuracy = correct / size
    print(f"{data_type.title()} Error: \n Accuracy: {current_accuracy:>0.1%}, Avg loss: {test_loss:>8f} \n")

    return current_accuracy, test_loss


def save_checkpoints(model, highest_accuracy_path, current_accuracy, highest_accuracy):
    if current_accuracy > highest_accuracy:

        torch.save(model, highest_accuracy_path)
        highest_accuracy = current_accuracy
        print(f'New record! Validation accuracy = {highest_accuracy:.1%}. Model is saved as: {highest_accuracy_path}\n')
    return highest_accuracy


def save_image(img_path, results_dir, prediction, classes):
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

@utilities.timer
def prediction(model_path, image_folder, task='prediction',
               batch_size=1, overwrite_results=True):
    model_path = pathlib.Path(model_path).expanduser().resolve()
    model = torch.load(model_path)  # TODO: 后续改为加载权重的模式。
    print(f'Using model: {model_path}')
    image_folder = pathlib.Path(image_folder).expanduser().resolve()

    if task == 'test_accuracy':
        # 预测时不要设置 dataset_type
        dataset = dataset_demo(image_folder, dataset_type='test')
        # Create data loaders.
        dataloader = DataLoader(dataset, batch_size=batch_size)

        loss_fn = torch.nn.CrossEntropyLoss()
        test(dataloader, model, loss_fn, data_type='test')
    elif task == 'prediction':
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
        model.eval()
        tic = time.perf_counter()
        t_preprocess = time.perf_counter()  # todo: 加上预处理和推理时间显示
        with torch.no_grad():
            # 如果要用 tqdm，就要知道文件的数量。但是因为 Path.iterdir() 没有 __len__ 方法，
            # 所以要用 os.listdir() 得到文件数量。
            images_quantity = len(os.listdir(image_folder))
            images_iterator = tqdm(image_folder.iterdir(), total=images_quantity, ncols=80)
            for each_image in images_iterator:
                if each_image.is_file():
                    # read_image 要求输入的为字符串类型，因此用 str 转换一下。
                    # pil_image 是一个 uint8 图片张量，形状为 (depth, height, width)。
                    pil_image = torchvision.io.read_image(str(each_image))

                    # pil_image 形状为 (depth, height, width)，数据类型为 float32，
                    # 并且数值被转换为 [0, 1]。
                    pil_image = torchvision.transforms.ConvertImageDtype(torch.float32)(pil_image)
                    # pil_image 的大小必须和模型要求的输入保持一致。
                    # 设置 antialias=True 是为了避免错误信息的提示。
                    pil_image = torchvision.transforms.Resize(
                        (300, 300), antialias=True)(pil_image)
                    pil_image = pil_image[None, ...]  # 预测时，需要输入 4D 张量

                    pil_image = pil_image.to(device)  # 移到 GPU/CPU 上。
                    pred = model(pil_image)
                    save_image(img_path=each_image, results_dir=results_dir,
                               prediction=pred, classes=classes)
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


def get_data(data_root=None, batch_size=8, get_class_id_only=False):
    if data_root is None:
        data_root = pathlib.Path(r'dataset_demo').expanduser().resolve()
    data_root = pathlib.Path(data_root).expanduser().resolve()
    # root = r'/home/leapting/work/cv/2023_05_06_datasets/kaggle_train_test'

    test_data = dataset_demo(data_root, dataset_type='test')
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    print(f'Checking classes: {test_data.classes}')

    if get_class_id_only:
        output = test_data.classes
    else:
        training_data = dataset_demo(data_root, dataset_type='train')
        validation_data = dataset_demo(data_root, dataset_type='validation')
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


def dataset_demo(root: pathlib.Path, dataset_type=None):
    # root = pathlib.Path(root)
    if dataset_type is not None:
        dataset_path = root / dataset_type  # dataset_type 是 'train'，'test' 等
    else:
        dataset_path = root
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                        std=[0.229, 0.224, 0.225])
    dataset = torchvision.datasets.ImageFolder(
        root=dataset_path, transform=transforms.Compose([
                transforms.Resize((300, 300)),  # 图片改为 300x300 大小
                transforms.ToTensor(),
                # normalize,  # 待定
            ]))

    # print(f'dataset: {type(dataset)}')
    # print(f'dataset.classes: {dataset.classes}')
    # print(f'dataset.class_to_idx: {dataset.class_to_idx}')
    # print(f'dataset.imgs: {dataset.imgs}')
    # show_dataset(dataset)

    return dataset


@utilities.timer
def main(data_root, lr=1.1e-2, batch_size=8, epochs=5, classes=2,
         checkpoint_path=None, save_model=True):
    """创建一个 RegNet 模型，并且对其进行训练。

    Arguments:
        data_root (str): 一个字符串，指向一个文件夹，其中放着训练集，验证集和测试集 3 个文件夹。
        lr (float): 一个浮点数，是模型的学习率。
        batch_size (int): 一个整数，是批量的大小。
        epochs (int): 一个整数，是训练的迭代次数。
        classes (int): 一个整数，是模型需要区分的类别数量。目前为 2 类，即只区分有异物和无异物。
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

    writer = SummaryWriter()

    highest_accuracy = 0  # 记录最高准确度，据此保存模型。

    print(f'{lr=:.2g}, {batch_size=}, {highest_accuracy=}')  # .2g 为自动使用 f 或 e

    # input_shape = (3, 32, 32)
    # size = np.prod(input_shape)
    model = ForeignMatterClassifier(classes).to(device)  # 注意把模型移到 device 上
    # print(model)

    # Create data loaders.
    train_dataloader, validation_dataloader, test_dataloader = get_data(data_root, batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 时间格式 '20230512_1016'，前半部分为日期，后半部分为小时和分钟
    time_suffix = time.strftime('%Y%m%d_%H%M')
    model_name = f'model_{time_suffix}.pt'
    highest_accuracy_path = checkpoint_path / model_name
    for t in range(epochs):
        print(f"Epoch {t + 1}/{epochs}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
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
    test_accuracy, _ = test(test_dataloader, best_model, loss_fn, data_type='test')
    print(f"Best test accuracy: {test_accuracy:.1%}")
    if save_model:
        # 加上测试集准确度之后再保存一次模型。
        add_accuracy = highest_accuracy_path.parent / (
                highest_accuracy_path.stem + f'_testacc_{test_accuracy * 1000:.0f}.pt')
        print(f'{add_accuracy=}')
        highest_accuracy_path.rename(add_accuracy)
        print(f'Best model is saved as: {highest_accuracy_path}\n')

    print("Done!")


if __name__ == '__main__':
    data_root = '~/work/cv/2023_05_04_regnet/classification_data'
    main(data_root=data_root, lr=1.1e-2, batch_size=8, epochs=2)

    # get_data(5)
    # dataset_demo()

    # 对文件夹内图片进行预测
    # model_path = r'checkpoints/model_20230516_1713.pt'
    # # image_folder = r'~/work/cv/2023_05_04_regnet/classification_data/test'
    # image_folder = r'prediction_images'
    # prediction(model_path=model_path, image_folder=image_folder,
    #            # task='test_accuracy',   # prediction
    #            batch_size=8)

    # inputs = torch.randn(size=(1, 3, 224, 224))
    # classifier_solar_panel = ForeignMatterClassifier()
    # print(f'classifier:\n{classifier_solar_panel}')
    # train_nodes, eval_nodes = get_graph_node_names(classifier_solar_panel)
    # print(f'train_nodes: \n{train_nodes}')

    # torchview.draw_graph(classifier_solar_panel, input_size=(1, 3, 224, 224),
    #                      save_graph=True, expand_nested=True,
    #                      filename='classifier_solar_panel',
    #                      device='meta')

    # get_and_extract_model()  # 查看 regnet_y_16gf 原始模型。

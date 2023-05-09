#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""该模块用于训练一个 RegNet 模型，对光伏板进行分类。主要区分光伏板上有异物和
没有异物两种情况。

版本号： 0.1
如需帮助，可联系 jun.liu@leapting.com
"""
import pathlib
import unicodedata

import graphviz
import torch
import torchview
from torchvision import transforms
from torch.utils.data import DataLoader

import torchvision.models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

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

        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    current_accuracy = correct / size
    print(f"Test Error: \n Accuracy: {current_accuracy:>0.1%}, Avg loss: {test_loss:>8f} \n")

    return current_accuracy


def save_checkpoints(model, model_path, current_accuracy, highest_accuracy):
    if current_accuracy > highest_accuracy:
        torch.save(model, model_path / 'model.pt')
        highest_accuracy = current_accuracy
        print(f'New record! {highest_accuracy=:.1%}. Model is saved.')
    return highest_accuracy


def prediction(model_folder, image_folder, dataset_type=None, batch_size=1):
    model_path = pathlib.Path(model_folder)
    model = torch.load(model_path / 'model.pt')  # TODO: 后续改为加载权重的模式。

    if dataset_type == 'test':
        # 预测时不要设置 dataset_type
        dataset = dataset_demo(image_folder, dataset_type=dataset_type)
        # Create data loaders.
        dataloader = DataLoader(dataset, batch_size=batch_size)

        loss_fn = torch.nn.CrossEntropyLoss()
        test(dataloader, model, loss_fn)
    else:
        classes = get_data(get_class_id_only=True)
        print(f'{classes = }')
        model.eval()
        image_folder = pathlib.Path(image_folder)
        with torch.no_grad():
            for each_image in image_folder.iterdir():
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
                probability = torch.sigmoid(pred)  # 将预测结果转换为概率值。
                class_idx = torch.argmax(probability)
                # print(f'{each_image.stem}, {probability.shape}, {probability}')
                print(f'{each_image.stem}, \t'
                      f'class: {classes[class_idx]}, '  # noqa
                      f'probability: {probability[0, class_idx]:.1%}')


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


def get_data(batch_size=8, get_class_id_only=False):
    root = r'dataset_demo'
    # root = r'/home/leapting/work/cv/2023_05_06_datasets/kaggle_train_test'

    test_data = dataset_demo(root, dataset_type='test')
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    if get_class_id_only:
        output = test_data.classes
    else:
        training_data = dataset_demo(root, dataset_type='train')
        # Create data loaders.
        # 因为自定义的光伏板数据集是固定放在 2 个文件夹，加载时必须打乱顺序，避免
        # 前一半全是 A 类别，后一半全是 B 类别的情况。如果不打乱，会使得前面几轮
        # 训练效果较差。
        train_dataloader = DataLoader(training_data,
                                      batch_size=batch_size, shuffle=True)
        output = train_dataloader, test_dataloader

    # show_dataset(train_dataloader, 3)  # 查看数据集内部。

    return output


def dataset_demo(root, dataset_type=None):
    root = pathlib.Path(root)
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
def main(lr=1e-3, batch_size=64, epochs=5, classes=2,
         model_path=None, save_model=True):
    # 保存到 checkpoints
    if model_path is None:
        model_path = pathlib.Path(__file__).parent / 'checkpoints'
    else:
        model_path = pathlib.Path(model_path) / 'checkpoints'
    if not model_path.exists():
        model_path.mkdir()
    print(f'Model will be saved to: {model_path}')
    highest_accuracy = 0  # 记录最高准确度，据此保存模型。

    print(f'{lr=}, {batch_size=}, {highest_accuracy=}')

    # input_shape = (3, 32, 32)
    # size = np.prod(input_shape)
    model = ForeignMatterClassifier(classes).to(device)  # 注意把模型移到 device 上
    # print(model)

    # Create data loaders.
    train_dataloader, test_dataloader = get_data(batch_size)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        current_accuracy = test(test_dataloader, model, loss_fn)
        if save_model:
            highest_accuracy = save_checkpoints(
                model=model, model_path=model_path,
                current_accuracy=current_accuracy,
                highest_accuracy=highest_accuracy)
    print("Done!")


if __name__ == '__main__':
    # main(lr=8e-3, batch_size=16, epochs=5,  classes=2)
    # get_data(5)
    # dataset_demo()

    model_folder = r'checkpoints'
    image_folder = r'prediction_images'
    prediction(model_folder=model_folder, image_folder=image_folder,
               # dataset_type='test',
               batch_size=8)

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

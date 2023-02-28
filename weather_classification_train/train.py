"""
需要修改的地方
21行：classname=? 按照自己的类别数修改
95行：路径为自己要保存的模型文件，
"""
from resnet import ResNet, BasicBlock, Bottleneck
import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.modules
from read_batch_data import train_data_loader, test_data_loader
import matplotlib.pyplot as plt
# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set hyperparameter
EPOCH = 30
LR = 0.001

# 网络层数可选18：[2,2,2,2],34:[3,4,6,3](每个块卷二次),50:[3,4,6,3](每个块卷三次),...
net = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2)  # block 为残差块，网络层，类别数，
net = net.to(device)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)  # sgd 优化器，

x_axis_train = []  # 保存训练的次数step
loss_train = []  # 保存每一步的loss
Acc_on_train = []  # 每一步训练的准确率
for epoch in range(EPOCH):   # epoch 训练次数
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(train_data_loader, 0):  # 共1577张图片，一个batch10张，共取了158次
        # prepare dataset
        length = len(train_data_loader)
        inputs, labels = data
        # labels.scatter_(1, labels, 1)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # forward & backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)    # 交叉熵计算损失
        loss.backward()    # 反向传播计算各个参数的梯度
        optimizer.step()    # 优化器更新梯度

        # print ac & loss in each batch
        sum_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)  # predicted是第1轴上最大值的索引
        total += labels.size(0)
        # print("预测结果为：", predicted.data, "实际结果为：", labels.data)
        correct += predicted.eq(labels.data).cpu().sum()  # eq():相等返回1否则返回0
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))

        x_axis_train.append(i + 1 + epoch * length)  # 将每个step索引保存在x_axis_train中
        loss_train.append(loss.item())  # 将每一步的loss保存在loss_train中
        Acc_on_train.append(100. * correct / total)  # 将每一步训练集的准确率保存下来
    # plt.figure(figsize=(20, 8), dpi=120)  # 初始化一个图，用于绘制下面需要的曲线
    # plt.subplot(1, 2, 1)  # 设置子图1
    # plt.xlabel('Step')  # 设置子图1 X轴的名称
    # plt.ylabel('Loss')  # 设置子图1 Y轴的名称
    # plt.title('Step_Loss curve')  # 设置子图1的名称
    # plt.plot(x_axis_train, loss_train, color="r", linestyle="--")  # 绘制step_loss曲线
    # plt.subplot(1, 2, 2)  # 设置子图2
    # plt.xlabel('Step')
    # plt.ylabel('Accuracy')
    # plt.title('Step_Acc curve')
    # plt.plot(x_axis_train, Acc_on_train, color="b", linestyle="--")  # 绘制step_Acc曲线
    # plt.show()  # 显示step_loss曲线
    print('Waiting Test...')
    with torch.no_grad():  # 测试的时候不需要计算梯度，所以测试放在该上下文管理器下
        correct = 0
        total = 0
        for data in test_data_loader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            # print("预测的概率值=", outputs.data)
            # print("预测结果为：", predicted.data, "实际结果为：", labels.data)
            # correct += (predicted == labels).sum()
        print("Total=====", total)
        print("100*correct",correct)
        print('total',total)
        print('Test\'s ac is: %.3f%%' % (100 * correct / total))

        # 当测试集准确率达标后，将模型保存下来
        if(correct / total) >= 0.3:
            torch.save(net.state_dict(), "/home/guoningwang/分类/classify/model_saved/resnet.pth",
                       _use_new_zipfile_serialization=False)
            print("--------------------------Model saved---------------------------")


print('Train has finished, total epoch is %d' % EPOCH)

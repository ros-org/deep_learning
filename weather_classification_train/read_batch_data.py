from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import cv2

# 过滤警告信息
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):  # 继承Dataset
    def __init__(self, base_dir, transform=None):  # 初始化一些属性
        self.base_dir = base_dir  # 文件路径
        self.transform = transform  # 对图形进行处理，如标准化、截取、转换等
        self.images_dir = os.listdir(self.base_dir)
        self.images_path_list = []  # 所有图片的路径列表
        self.labels_list = []  # 所有图片对应的标签列表

        for i in self.images_dir:
            image_dir_path = os.path.join(self.base_dir, i)

            for j in os.listdir(image_dir_path):
                image_path = os.path.join(image_dir_path, j)
                self.images_path_list.append(image_path)
                if "categories0" in image_path:
                    self.labels_list.append(0)
                elif "categories1" in image_path:
                    self.labels_list.append(1)
                elif "categories2" in image_path:
                    self.labels_list.append(2)
                elif "categories3" in image_path:
                    self.labels_list.append(3)


                elif "categories4" in image_path:
                    self.labels_list.append(4)
                # else:
                #     self.labels_list.append(5)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.images_path_list)

    def getImagePath(self, Index):
        return self.images_path_list[Index]

    def __getitem__(self, index):  # 根据索引index返回图像及标签
        image = cv2.imread(self.images_path_list[index], cv2.IMREAD_UNCHANGED)  # 使用CV2读取数据，返回的是ndarray类型数据
        # print("Image path:", self.images_path_list[index])
        label = self.labels_list[index]
        # print("label=", label)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# 图像扩充
image_train_transformer = transforms.Compose([             #transforms.Compose 是pytorch 的图像预处理包，Compose 是把多个步骤合到一起
    transforms.ToTensor(),                 # (H,W,C)-->(C,H,W),[0,255]-->[0.0,1.0]
    transforms.Resize([112, 112]),         # 把给定的图片Resize 到112，112
    transforms.RandomHorizontalFlip(),     # 以0.5 的概率水平翻转给定的 pil 图像，
    transforms.Normalize(mean=0, std=1)])  # 单通道图片， 用均值和标准差归一化张量图像

image_test_transformer = transforms.Compose([   # 测试集图像处理
    transforms.ToTensor(),
    transforms.Resize([112, 112]),
    transforms.Normalize(mean=0, std=1)])  # 单通道图片

base_dir_train = "/home/guoningwang/分类/classify/datasets/4_categories/train"
base_dir_test = "/home/guoningwang/分类/classify/datasets/4_categories/test"
train_dataset = MyDataset(base_dir_train, transform=image_train_transformer)
test_dataset = MyDataset(base_dir_test, transform=image_test_transformer)
testDataset_total_num = test_dataset.__len__()  # 测试集数量总共有多少
train_data_loader = DataLoader(train_dataset, 50, shuffle=True)
# test_data_loader = DataLoader(test_dataset, testDataset_total_num, batch_size=1,shuffle=False)
test_data_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)





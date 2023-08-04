# ==========================================================
# 功能  ：将检测的txt文件中的box框坐标按要求写入到检测json文件中；当前该写好的json文件并不包含base64格式的图片数据；
#         我已经实现了base64和图片互转的函数，如果有需要请自己进行转换并将转换好的数据写到适当的位置；
# 注意：需要将图片转成png格式
# 文件名  ：transDetectionTxt2DetectionJson.py
# 相关文件：无
# 作者    ：Liangliang Bai (liangliang.bai@leapting.com)
# 版权    ：<Copyright(C) 2022- Hu Zhou leapting Technology Co., Ltd. All rights reserved.>
# 修改记录：
# 日  期        版本     修改人   走读人  修改记录
#
# 2023.08.03    1.0.0.1  白亮亮
# ==========================================================

import os
import json
import base64


label_name = ["_background_", "abnormal"]


# 图片转base64
def trans_image_to_base64(image_path):
    if not os.path.exists(image_path):
        raise Exception("图片%s不存在，请检查！" % image_path)
    else:
        with open(image_path, "rb") as f:               # 转为二进制格式
            base64_data = base64.b64encode(f.read())    # 使用base64进行加密
            if isinstance(base64_data, bytes):
                return str(base64_data, encoding='utf-8')


# base64转图片
def trans_base64_to_image(base64_buffer, image_path=None):
    image_data = base64.b64decode(base64_buffer)
    if len(image_path) != 0:
        with open(image_path, 'wb') as image_file:
            image_file.write(image_data)
    else:
        return image_data


# 输入txt路径，返回框数及每个框的四个坐标值、图片名称
# 返回txt文件中的内容，以二维数组的方式返回：[[图片名],[图片高，宽],[x1,y1,x2,y2],...],
# 二维数组前两个元素存的不是框坐标，所以框个数是返回的列表长度-2；
def getBoxCoord(txtPath, imageData):
    data = []
    f = open(txtPath)
    for line in f.readlines():
        array = []
        line = line.strip()
        line = line.split(' ')
        for item in line:
            array.append(item)
        data.append(array)
    data.append(imageData)
    return data


# 生成单个框的字典
def genBoxDict(label_name, x1, y1, x2, y2):
    box_dict = {
        'label': label_name,
        'points': [
            [
                float(x1),
                float(y1)
            ],
            [
                float(x2),
                float(y2)
            ]
        ],
        'group_id': None,
        'shape_type': "rectangle",
        'flags': {}
    }
    return box_dict


# 生成包含所有框的字典
def genBoxesDict(data, points_num):
    boxes_dict = []
    for i in range(points_num):
        point_dict = genBoxDict(label_name[int(data[i+2][0])], data[i+2][1], data[i+2][2], data[i+2][3], data[i+2][4])
        boxes_dict.append(point_dict)
    return boxes_dict


# 生成一个图片标签的dict，用于写入json
def genDict(points_dict, data):
    data_to_write = {
        'version': "4.5.9",
        'flags': {},
        'shapes': points_dict,
        'imagePath': data[0][0],
        'imageData': data[-1],
        'imageHeight': int(data[1][1]),
        'imageWidth': int(data[1][0])
    }
    return data_to_write


# ==============================》将数据写到指定的json中《============================= #
# Parameters
#     Input  Params:
#         data_to_write:待写入到新json文件的数据
#         json_file_path:新json文件的路径
#     Output params:
#         None
#     Return values:
#         None
# ==============================》将数据写到指定的json中《============================= #
def writeDataToJson(data_to_write, json_file_path):
    json_str = json.dumps(data_to_write, indent=4)
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_str)


def txt_to_json(txt_base_dir, image_base_dir, json_dir):
    if not os.path.exists(json_dir):  # 判断所在目录下是否有该文件名的文件夹
        os.mkdir(json_dir)

    file_names = os.listdir(txt_base_dir)
    for i, file_name in enumerate(file_names):
        print("file_name=", file_name)
        file_name_dir = txt_base_dir+file_name
        imageData = trans_image_to_base64((image_base_dir + file_name).replace(".txt", ".png"))  # 将图转为base64
        data = getBoxCoord(file_name_dir, imageData)
        boxes_num = len(data)-3
        print("boxes_num=", boxes_num)
        boxes_dict = genBoxesDict(data, boxes_num)
        data_to_write = genDict(boxes_dict, data)
        image_name = file_name.split('.')[0]
        print("图片索引:", i, ",The name of current image is:", image_name)
        jsonFileName = json_dir + image_name + ".json"
        writeDataToJson(data_to_write, jsonFileName)


def main():
    base_dir = r"D:\LT_DATASETS_PATROL_ROBOT\segmentation\train/"

    # 待转换的txt文件所在文件夹
    txt_base_dir = base_dir + "txt_aug/"

    # 待转base64的图片所在文件夹
    image_base_dir = base_dir + "image_aug/"

    # 转换好的json文件存放文件夹
    json_dir = base_dir + "/detection_json/"
    txt_to_json(txt_base_dir, image_base_dir, json_dir)


if __name__ == '__main__':
    main()



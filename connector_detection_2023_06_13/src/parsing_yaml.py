#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本模块用于从 YOLOv8 的数据文件 .yaml 中，提取类别名字和类别 id。
该 yaml 文件必须以类别名字和 ID 部分结尾，从 names 开始，包含 id 和
类别名字对应的部分。结尾部分的形式如下示例：

names:
  0: disconnected  # 区分断开和未断开的接插件
  1: connected
  ...

版本号： 0.1
日期： 2023-06-15
作者： jun.liu@leapting.com
"""


def parsing_yaml(yaml_path):
    start_index = False
    category_id_mapping = {}
    with open(yaml_path, 'r') as f:
        for line in f:
            if start_index:
                split_line = line.split()
                category_id = split_line[0].rstrip(':')
                category = split_line[1]
                category_id_mapping[category] = int(category_id)
            if line == 'names:\n':  # open 函数把换行符被统一转换为 \n
                start_index = True
    return category_id_mapping


if __name__ == '__main__':
    yaml_path = r'connector.yaml'
    parsing_yaml(yaml_path)

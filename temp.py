#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


# 读取数据
def initDataForBert(path, debug=False):
    print("In initDataForBert function of auto_absa_dataProcess.py...")

    file_names = ["Big", "Medium", "MediumBig", "Micro", "Small"]

    # 将所有文件内容读取出来，生成一份数据
    data = produceAllData(path, file_names)

    data = pd.read_csv(path)
    if debug:
        data = data[:200]

    length = len(data)
    # data = data[:int(length / 3000)]

    # data = data[:50]
    y = data[['location', 'service', 'price', 'environment', 'dish']]
    print("y = ", y.head())

    # 对原评论文本进行清洗，去回车符 去空格

    # y_cols_name = data.columns.values.tolist()[2:22]
    y_cols_name = data.columns.values.tolist()[2:7]
    print(">>>data = ", data.head())
    print(">>>data'type = ", type(data))
    print(">>>data's shape = ", data.shape)
    print(">>>y_cols_name = ", y_cols_name)

    # print("end of initData function in dataProcess.py...")

    data_content = data['content']
    # print("data_content's head = ", data_content.head())

    '''
    print("data_content's type = ", type(data_content))
    for content in data_content:
        print("content = ", content)
    '''

    return data_content, y_cols_name, y


# 读取所有文件内容，生成一份数据
def produceAllData(path, file_names):
    data = []

    return data


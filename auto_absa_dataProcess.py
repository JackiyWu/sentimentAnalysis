#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from keras.utils import to_categorical

import auto_absa_config as config


# 读取数据,ratio表示分割比例，划分训练集验证集
def initDataForBert(path, ratio, debug=False):
    print("In initDataForBert function of auto_absa_dataProcess.py...")

    file_names = ["Big", "Medium", "MediumBig", "Micro", "Small"]

    # 将所有文件内容读取出来，生成一份数据
    data = produceAllData(path, file_names, debug)
    print("data's shape = ", data.shape)
    # print("data = ", data)
    # print(data.loc[[0]])
    print("*" * 50)

    # 对data进行shuffle
    data = data.sample(frac=1)

    # 索引重置
    data = data.reset_index(drop=True)
    # print("data = ", data)
    # print(data.loc[[0]])
    y = data[config.col_names]
    # print("y = ", y)

    # 生成训练集
    length = len(data)
    train_length = int(length * ratio)
    data_train = data[: train_length]
    # print("data_train pre = ", data_train)
    # print(data_train.loc[[0]])
    data_validation = data[train_length:]
    # print("data_validation pre = ", data_validation)
    # print(data_validation.loc[[0]])

    y_train = data_train[config.col_names]
    y_validation = data_validation[config.col_names]
    # print("y = ", y.head())

    data_train = data_train["content"]
    # print("data_train = ", data_train.head())
    data_validation = data_validation["content"]
    # print("data_validation = ", data_validation.head())

    # print("*" * 50)
    # print("data_train's shape = ", data_train.shape)
    # print("data_validation's shape = ", data_validation.shape)

    return data_train, config.col_names, y_train, data_validation, y_validation


# 读取所有文件内容，生成一份数据
def produceAllData(path, file_names, DEBUG=False):
    all_data = pd.DataFrame()

    for file_name in file_names:
        current_path = path + file_name + ".csv"
        current_data = produceOneData(current_path, DEBUG)
        # print("current_data = ", current_data)
        # print("*" * 50)
        print(file_name, "data's shape = ", current_data.shape)
        all_data = all_data.append(current_data)

    # print("all_data's columns = ", all_data.columns.values)
    # print("all_data = ", all_data)
    # print("*" * 50)

    return all_data


# 处理单个csv文件，生成指定格式
def produceOneData(path, DEBUG=False):
    data = pd.read_csv(path, names=config.all_names, header=0, encoding="utf-8")
    if DEBUG:
        data = data[:50]

    # 将所有评论内容为空的属性标签改为0
    # print("替换之前")
    # print(data[config.col_names])
    data = replaceValue(data)
    # print("替换之后")
    # print(data[config.col_names])

    # 生成一个content
    data["content"] = data["space"].map(str) + "。" + data["power"].map(str) + "。" + data["manipulation"].map(str) + "。" + data["consumption"].map(str) + "。" + \
                      data["comfort"].map(str) + "。" + data["outside"].map(str) + "。" + data["inside"].map(str) + "。" + data["value"].map(str)
    # print(data["content"].head())
    # print("data's columns = ", data.columns.values)
    # print("produceOneData data = ", data)
    # print("*" * 50)
    # 删除所有属性均为空的行
    data = data.drop(index=(data.loc[(data['content'] == '0。0。0。0。0。0。0。0')].index))

    return data


# 将所有评论内容为空的属性标签改为0
def replaceValue(data):
    data.loc[data['space'] == '0', ['space_label']] = 0
    data.loc[data['power'] == '0', ['power_label']] = 0
    data.loc[data['manipulation'] == '0', ['manipulation_label']] = 0
    data.loc[data['consumption'] == '0', ['consumption_label']] = 0
    data.loc[data['comfort'] == '0', ['comfort_label']] = 0
    data.loc[data['outside'] == '0', ['outside_label']] = 0
    data.loc[data['inside'] == '0', ['inside_label']] = 0
    data.loc[data['value'] == '0', ['value_label']] = 0

    return data


# 批量产生训练数据
def generateSetForBert(X_value, Y_value, batch_size, tokenizer):
    # print("This is generateSetForBert...")
    length = len(Y_value)

    while True:
        cnt = 0  # 记录当前是否够一个batch
        X1 = []
        X2 = []
        Y = []
        i = 0  # 记录Y的遍历
        cnt_Y = 0
        for line in X_value:
            x1, x2 = parseLineForBert(line, tokenizer)
            X1.append(x1)
            X2.append(x2)
            i += 1
            cnt += 1
            if cnt == batch_size or i == length:
                # print("cnt_Y's type = ", type(cnt_Y))
                # print("i's type = ", type(i))
                Y = Y_value[int(cnt_Y): int(i)]
                # print("Y = ", Y)
                cnt_Y += batch_size

                cnt = 0
                yield ([np.array(X1), np.array(X2)], to_categorical(Y, num_classes=6))
                X1 = []
                X2 = []
                Y = []


# 将text转为token
def parseLineForBert(line, tokenizer):
    indices, segments = tokenizer.encode(first=line, max_len=512)

    return np.array(indices), np.array(segments)


# 批量产生X
def generateXSetForBert(X_value, y_length, batch_size, tokenizer):
    while True:
        # print("in generateXSetForBert...")
        cnt = 0
        X1 = []
        X2 = []
        i = 0
        for line in X_value:
            x1, x2 = parseLineForBert(line, tokenizer)
            X1.append(x1)
            X2.append(x2)
            i += 1
            cnt += 1
            if cnt == batch_size or i == y_length:
                cnt = 0
                yield ([np.array(X1), np.array(X2)])
                X1 = []
                X2 = []

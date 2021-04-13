#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from keras.utils import to_categorical

import auto_absa_config as config


# 读取数据,ratio表示分割比例，划分训练集验证集
def initDataForBert(path, ratio, debug=False):
    print("In initDataForBert function of auto_absa_dataProcess.py...")

    file_names = ["Big", "Medium", "MediumBig", "Micro", "Small", "New"]

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
        data = data[:5]

    # 将所有评论内容为空的属性标签改为0
    data = replaceValue(data)

    # 修改标签：2→1,3→1,4→2,5→3，目前包括1、2、3
    data = replaceValue2(data)

    # 生成一个content,不带属性名称
    # data["content"] = data["space"].map(str) + data["power"].map(str) + data["manipulation"].map(str) + data["consumption"].map(str) + \
    #                   data["comfort"].map(str) + data["outside"].map(str) + data["inside"].map(str) + data["value"].map(str)

    # 生成一个content
    data["content"] = data["space"].map(str) + "。space。" + data["power"].map(str) + "。power。" + data["manipulation"].map(str) + "。manipulation。" + data["consumption"].map(str) + "。consumption。" + \
                      data["comfort"].map(str) + "。comfort。" + data["outside"].map(str) + "。outside。" + data["inside"].map(str) + "。inside。" + data["value"].map(str) + "。value。"

    # 删除所有属性均为空的行
    data = data.drop(index=data.loc[(data['content'] == '0。space。0。power。0。manipulation。0。consumption。0。comfort。0。outside。0。inside。0。value。')].index)

    # 删除所有属性均为3的行（已经删了）
    # data = data.drop(index=(data.loc[(data['space'] == 3) & (data['power'] == 3) & (data['manipulation'] == 3) & (data['consumption'] == 3)
    #                              & (data['comfort'] == 3) & (data['outside'] == 3) & (data['inside'] == 3) & (data['value'] == 3)].index))

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


# 修改标签：2→1,3→1,4→2,5→3，目前包括1、2、3
def replaceValue2(data):
    for col_name in config.col_names:
        # print("*" * 50)
        # print("1 Y = ", data[col_name])
        data.loc[data[col_name] == 2, [col_name]] = 1
        # print("2 Y = ", data[col_name])
        data.loc[data[col_name] == 3, [col_name]] = 1
        # print("3 Y = ", data[col_name])
        data.loc[data[col_name] == 4, [col_name]] = 2
        data.loc[data[col_name] == 5, [col_name]] = 3
        # print("4 Y = ", data[col_name])
        # print("*" * 50)

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
                yield ([np.array(X1), np.array(X2)], to_categorical(Y, num_classes=4))
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

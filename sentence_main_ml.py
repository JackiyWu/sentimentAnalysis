#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import numpy as np
from sklearn.model_selection import KFold

import sentence_model as model
import sentence_dataProcess as dp
import auto_usefulness_dataProcess as u_dp

debug = False

'''
使用机器学习进行情感分类
读取文本→去标点符号→分词→去停用词→读取腾讯词向量→文本向量化表示（求平均）→送入机器学习模型
'''

if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in sentence_main_ml.py ...")

    # 1.读取评论文本内容+预处理
    print("》》》开始读取文本数据。。。")
    # origin_data, X, Y = dp.readFromCSV(debug)
    origin_data = dp.readFromCSV2(debug)
    # print("X's length = ", len(X))
    # print("Y's length = ", len(Y))
    # print("Y = ", Y.tolist())
    # X = X.tolist()
    print("*" * 150)
    reviews = origin_data["reviews"]
    label = origin_data["label"]

    # 读取腾讯词向量+向量化表示X
    X_contents = np.array(dp.sentence2vector(np.array(reviews), debug))
    print("X_contents' type = ", type(X_contents))
    print("X_contents' shape = ", X_contents.shape)
    origin_data['padding'] = X_contents.tolist()
    print("*" * 150)

    # print("X's type = ", type(X))
    # print("Y's type = ", type(Y))

    training = origin_data.loc[origin_data['tag'] == 'training']
    # print("training['padding']'s type = ", type(training['padding']))
    X = np.array(list(training['padding']))
    Y = np.array(list(training['label']))
    training_corn = origin_data.loc[origin_data['tag'] == 'corn']
    X_contents_corn = np.array(list(training_corn['padding']))
    Y_corn = np.array(list(training_corn['label']))
    training_apple = origin_data.loc[origin_data['tag'] == 'apple']
    X_contents_apple = np.array(list(training_apple['padding']))
    Y_apple = np.array(list(training_apple['label']))

    print("X's type = ", type(X))
    # print("X_contents = ", X_contents)
    print("Y's type = ", type(Y))
    print("X's shape = ", X.shape)
    print("Y's shape = ", Y.shape)

    # model_names = ["svm"]
    model_names = ["bayes", "ada", "svm", "randomForest", "decisionTree", "logicRegression"]
    for model_name in model_names:
        print("当前模型为", model_name)
        # 交叉验证数据集
        kf = KFold(n_splits=10)
        current_k = 0
        for train_index, validation_index in kf.split(X):
            print("正在进行第", current_k, "轮交叉验证。。。")
            current_k += 1
            # print("train_index = ", train_index, ", validation_index = ", validation_index)
            X_train, Y_train = X[train_index], Y[train_index]
            X_validation, Y_validation = X[validation_index], Y[validation_index]

            # 3.构建&训练模型
            print("》》》开始构建&训练模型。。。")
            if model_name.startswith("bayes"):
                current_model = model.bayesModel()
            elif model_name.startswith("ada"):
                current_model = model.adaBoostModel()
            elif model_name.startswith("svm"):
                current_model = model.svmModel()
            elif model_name.startswith("logic"):
                current_model = model.logicRegressionModel()
            elif model_name.startswith("decision"):
                current_model = model.decisionTreeModel()
            elif model_name.startswith("random"):
                current_model = model.randomForestModel()

            current_model.fit(X_train, Y_train)

        # 4.预测结果
        # corn
        print("》》》开始预测结果。。。")
        predicts_corn = current_model.predict(X_contents_corn)
        # 5.计算各种评价指标&保存结果
        dp.calculateScore(Y_corn, predicts_corn, 'corn_' + model_name, debug)

        # apple
        print("》》》开始预测结果。。。")
        predicts_apple = current_model.predict(X_contents_apple)
        # 5.计算各种评价指标&保存结果
        dp.calculateScore(Y_apple, predicts_apple, 'apple_' + model_name, debug)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of auto_usefulness_main.py...")


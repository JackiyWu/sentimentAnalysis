#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import numpy as np
from sklearn.model_selection import KFold

import sentence_model as model
import sentence_dataProcess as dp
import auto_usefulness_dataProcess as u_dp

debug = True

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
    origin_data, X, Y = dp.readFromCSV(debug)
    print("X's length = ", len(X))
    print("Y's length = ", len(Y))
    # print("Y = ", Y.tolist())
    X = X.tolist()

    # 读取腾讯词向量+向量化表示X
    X = np.array(dp.sentence2vector(X, False))

    # print("X's type = ", type(X))
    # print("Y's type = ", type(Y))

    # model_names = ["svm"]
    model_names = ["bayes", "ada", "SVM", "randomForest", "decisionTree", "logicRegression"]
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
            elif model_name.startswith("decisionTree"):
                current_model = model.decisionTreeModel()
            elif model_name.startswith("random"):
                current_model = model.randomForestModel()

            current_model.fit(X_train, Y_train)

            # 4.预测结果
            print("》》》开始预测结果。。。")
            predicts = current_model.predict(X_validation)

            # 5.计算各种评价指标&保存结果
            dp.calculateScore(Y_validation, predicts, model_name, debug)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of auto_usefulness_main.py...")


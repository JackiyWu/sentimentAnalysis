#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import auto_usefulness_dataProcess as dp
import auto_usefulness_model as model

from sklearn.model_selection import KFold


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in auto_usefulness_main.py ...")

    # 1.读取数据+预处理
    print("》》》开始读取数据。。。")
    origin_data, X, Y = dp.readFromCSV(True)
    print("data's length = ", len(origin_data))
    print("X's type = ", type(X))
    print("Y's type = ", type(Y))

    # 2.划分数据集，交叉验证
    '''
    ratio = 0.8
    print("》》》划分数据集，比例为", ratio)
    X_train, Y_train, X_validation, Y_validation = dp.dataSplit(X, Y, ratio)
    '''

    model_names = ["svm"]
    # model_names = ["bayes", "ada", "SVM", "randomForest", "decisionTree", "logicRegression"]
    # model_name中带_0_的表示点赞数为0的标签是0，否则为1
    for model_name in model_names:
        model_name = model_name + "_0_0_"

        # 2.交叉验证数据集
        kf = KFold(n_splits=10)
        current_k = 0
        for train_index, validation_index in kf.split(X):
            print("正在进行第", current_k, "轮交叉验证。。。")
            current_k += 1
            print("train_index = ", train_index, ", validation_index = ", validation_index)
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
            dp.calculateScore(Y_validation, predicts, model_name)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of auto_usefulness_main.py...")


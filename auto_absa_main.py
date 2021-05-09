#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import auto_absa_dataProcess as dp
import auto_absa_models
import auto_absa_config as config

debug = False

if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in absa_main.py ...")

    # 1.读取文件data_train, config.col_names, y_train, data_validation, y_validation
    print("》》》【1】正在读取文件", "。" * 100)
    # dp.initDataForBert(config.perception_data)
    ratio = 0.8
    X, y_cols, Y, X_validation, Y_validation = dp.initDataForBert(config.perception_data, ratio, debug)
    print("X.shape = ", X.shape)
    print("Y.shape = ", Y.shape)
    print("X_validation.shape = ", X_validation.shape)
    print("Y_validation.shape = ", Y_validation.shape)
    # print("X = ", X)
    # print("Y = ", Y.head())
    # print("X_validation = ", X_validation)
    # print("y_validation = ", Y_validation.head())

    # 2.加载tokenizer
    print("》》》【2】正在加载tokenizer", "。" * 100)
    tokenizer = auto_absa_models.get_tokenizer()

    # 3.模型参数设置
    if debug:
        epochs = [1]
    else:
        epochs = [3]
    batch_sizes = [80]
    batch_size_validation = 600
    times = 1
    print("training times = ", times)

    model_name = "BertCNNBiGRUModel"

    # 4.模型构建
    print("》》》【4】正在构建模型", "。" * 100)
    if model_name.startswith("BertCNNModel"):
        filters = [128]
        window_sizes = [4]
        for cnn_filter in filters:
            for window_size in window_sizes:
                for batch_size in batch_sizes:
                    for epoch in epochs:
                        experiment_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_filter_" + str(cnn_filter) + "_windowSize_" + str(window_size)
                        print("experiment_name = ", experiment_name)
                        for i in range(times):
                            model = auto_absa_models.createBertCNNModel(cnn_filter, window_size)
                            auto_absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation)
    elif model_name.startswith("BertGRUModel"):
        gru_output_dim_1 = [256]
        for dim_1 in gru_output_dim_1:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    experiment_name = model_name + "_gru_dim_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                    print("experiment_name = ", experiment_name)
                    for i in range(times):
                        model = auto_absa_models.createBertGRUModel(dim_1)
                        auto_absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation)
    elif model_name.startswith("BertCNNBiGRUModel"):
        filters = [128]
        window_sizes = [4]
        gru_output_dim_1 = [256]
        for cnn_filter in filters:
            for window_size in window_sizes:
                for dim_1 in gru_output_dim_1:
                    for batch_size in batch_sizes:
                        for epoch in epochs:
                            experiment_name = model_name + "_filter_" + str(cnn_filter) + "_window_size_" + str(window_size) + "_gru_dim_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                            print("experiment_name = ", experiment_name)
                            for i in range(times):
                                print("current times is ", i)
                                model = auto_absa_models.createBertCNNBiGRUModel(cnn_filter, window_size, dim_1)
                                auto_absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, debug)
    elif model_name.startswith("BertLSTMModel"):
        dims_1 = [64]
        for dim_1 in dims_1:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    experiment_name = model_name + "_dim1_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                    print("experiment_name = ", experiment_name)
                    for i in range(times):
                        model = auto_absa_models.createBertLSTMModel(dim_1)
                        auto_absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation)
    else:
        print(">>>模型名称有问题哦！！！")

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of absa_main.py...")

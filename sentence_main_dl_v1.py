#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import numpy as np

import sentence_model as model
import sentence_dataProcess as dp
import auto_usefulness_dataProcess as u_dp

debug = False

if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in sentence_main_dl.py ...")

    # 1.读取评论文本内容+预处理
    print("》》》开始读取文本数据。。。")
    origin_data, X, Y = dp.readFromCSV(debug)
    # origin_data = dp.readFromCSV2(debug)
    # print("X = ", X.tolist())
    # print("Y = ", Y.tolist())
    # print("origin_data.head = ", origin_data.head())
    print("*" * 150)
    reviews = origin_data["reviews"]
    label = origin_data["label"]
    # print(reviews)
    # print(label)

    X_contents, word_index, maxlen = u_dp.contentsPadding(X)
    print("maxlen = ", maxlen)
    print("X_contents' type = ", type(X_contents))
    # print("X_contents = ", X_contents)
    print("*" * 150)
    print("X_contents' shape = ", X_contents.shape)
    origin_data['padding'] = X_contents.tolist()
    # print("origin_data.head = ", origin_data.head())
    print("*" * 150)

    # X_corn = origin_data.loc['corn', 'padding']
    # X_corn = origin_data.loc[origin_data['tag'] == 'corn']
    # print("X_corn.head = ", X_corn.head())
    # X_corn_padding = X_corn['padding']
    # print("X_corn_padding = ", X_corn_padding)

    '''
    print("*" * 150)
    sys.exit(0)
    '''

    # 2.读取腾讯词向量
    print("》》》读取腾讯词向量。。。")
    if debug:
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
    else:
        embedding_matrix = model.load_word2vec(word_index)
    print("embedding_matrix's shape = ", embedding_matrix.shape)

    dict_length = min(150000, len(word_index) + 1)
    print("dict_length = ", dict_length)

    '''
    print("X_contents' length = ", len(X_contents))
    print("Y' length = ", len(Y))
    if len(X_contents) != len(Y):
        print("长度不一致！！！")
        sys.exit(-1)
    '''

    # 5.构建深度学习模型
    print("》》》构建深度学习模型。。。")
    batch_size = 256
    # epochs = [4, 5, 6]

    # model_names = ["cnnbilstm"]
    model_names = ["cnn", "lstm", "gru", "mlp", "bilstm", "bigru", "cnnbilstm"]
    epoch = 4
    for model_name in model_names:
        print("current model is", model_name)

        if model_name == "cnn":
            cnn_nodes = [64]
            epochs = [4]
            for epoch in epochs:
                for cnn_node in cnn_nodes:
                    model_name = "cnn_epoch_" + str(epoch) + "_node_" + str(cnn_node)
                    current_model = model.createCNNModel(maxlen, dict_length, embedding_matrix, cnn_node)
                    # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
                    model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
        elif model_name == "lstm":
            dim = 64
            current_model = model.createLSTMModel(maxlen, dict_length, embedding_matrix, dim)
            # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
            model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
        elif model_name == "gru":
            dim = 64
            current_model = model.createGRUModel(maxlen, dict_length, embedding_matrix, dim)
            print("》》》训练深度学习模型。。。")
            # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
            model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
        elif model_name == "mlp":
            epochs = [5]
            for epoch in epochs:
                model_name = "mlp_epoch_" + str(epoch)
                current_model = model.createMLPModel(maxlen, dict_length, embedding_matrix, 64)
                print("》》》训练深度学习模型。。。")
                # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
                model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
                '''
                model.trainV2Model(model_name, maxlen, dict_length, embedding_matrix, X_contents, Y, epoch, batch_size, debug)
                '''
        elif model_name == "bilstm":
            bi_flag = True
            dim = 64
            current_model = model.createLSTMModel(maxlen, dict_length, embedding_matrix, dim, bi_flag)
            # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
            model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
        elif model_name == "bigru":
            bi_flag = True
            dim = 64
            current_model = model.createGRUModel(maxlen, dict_length, embedding_matrix, dim, bi_flag)
            print("》》》训练深度学习模型。。。")
            # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
            model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
        elif model_name == "cnnbigru":
            # cnn_filters = [64]
            cnn_filters = [8, 16, 32, 64, 128, 256, 512]
            window_sizes = [5]
            # window_sizes = [2, 3, 4, 5, 6, 7, 8]
            gru_output_dims = [512]
            # gru_output_dims = [8, 16, 32, 64, 128, 256, 512]
            for cnn_filter in cnn_filters:
                for window_size in window_sizes:
                    for gru in gru_output_dims:
                        model_name = "cnnbigru_filter_" + str(cnn_filter) + "_window_" + str(window_size) + "_dim_" + str(gru)
                        current_model = model.createCNNBiGRUModel(maxlen, dict_length, embedding_matrix, cnn_filter, window_size, gru)
                        print("》》》训练深度学习模型。。。")
                        # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
                        model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
        elif model_name == "cnnbilstm":
            cnn_filters = [64]
            window_sizes = [4]
            gru_output_dims = [32]
            for cnn_filter in cnn_filters:
                for window_size in window_sizes:
                    for dim in gru_output_dims:
                        model_name = "cnnbilstm_filter_" + str(cnn_filter) + "_window_" + str(window_size) + "_dim_" + str(dim)
                        current_model = model.createCNNBiLSTMModel(maxlen, dict_length, embedding_matrix, cnn_filter, window_size, dim)
                        print("》》》训练深度学习模型。。。")
                        # model.trainModel(model_name, current_model, origin_data, epoch, batch_size, debug)
                        model.trainModelV1(model_name, current_model, X_contents, Y, epoch, batch_size, debug)
        else:
            print("模型名字有问题！！！！！")

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of auto_usefulness_main.py...")


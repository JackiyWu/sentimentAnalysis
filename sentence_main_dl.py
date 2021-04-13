#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import numpy as np

import sentence_model as model
import sentence_dataProcess as dp
import auto_usefulness_dataProcess as u_dp

debug = True

if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in sentence_main_dl.py ...")

    # 1.读取评论文本内容+预处理
    print("》》》开始读取文本数据。。。")
    origin_data, X, Y = dp.readFromCSV(debug)
    print("X = ", X.tolist())
    print("Y = ", Y.tolist())

    X_contents, word_index, maxlen = u_dp.contentsPadding(X)

    # 2.读取腾讯词向量
    print("》》》读取腾讯词向量。。。")
    if debug:
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
    else:
        embedding_matrix = model.load_word2vec(word_index)
    print("embedding_matrix's shape = ", embedding_matrix.shape)

    dict_length = min(150000, len(word_index) + 1)
    print("dict_length = ", dict_length)

    if len(X_contents) != len(Y):
        print("X_contents' length = ", len(X_contents))
        print("Y' length = ", len(Y))
        print("长度不一致！！！")
        sys.exit(-1)

    # 5.构建深度学习模型
    print("》》》构建深度学习模型。。。")
    batch_size = 256
    epochs = [4, 5, 6]

    model_name = "cnnbilstm"
    print("current model is", model_name)

    if model_name == "cnnbigru":
        cnn_nodes = [32]
        current_model = model.createCNNModel(maxlen, dict_length, embedding_matrix)
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    elif model_name == "lstm":
        dim = 64
        current_model = model.createLSTMModel(maxlen, dict_length, embedding_matrix, dim)
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    elif model_name == "gru":
        dim = 64
        current_model = model.createGRUModel(maxlen, dict_length, embedding_matrix, dim)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    elif model_name == "mlp":
        current_model = model.createMLPModel(maxlen, dict_length, embedding_matrix, 256)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    elif model_name == "bilstm":
        bi_flag = True
        dim = 64
        current_model = model.createLSTMModel(maxlen, dict_length, embedding_matrix, dim, bi_flag)
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    elif model_name == "bigru":
        bi_flag = True
        dim = 64
        current_model = model.createGRUModel(maxlen, dict_length, embedding_matrix, dim, bi_flag)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    elif model_name == "cnnbigru":
        cnn_filter = 64
        window_size = 4
        gru_output_dim = 32
        current_model = model.createCNNBiGRUModel(maxlen, dict_length, embedding_matrix, cnn_filter, window_size, gru_output_dim)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    elif model_name == "cnnbilstm":
        cnn_filter = 64
        window_size = 4
        dim = 32
        current_model = model.createCNNBiLSTMModel(maxlen, dict_length, embedding_matrix, cnn_filter, window_size, dim)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_contents, Y, 3, batch_size, debug)
    else:
        print("模型名字有问题！！！！！")

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of auto_usefulness_main.py...")


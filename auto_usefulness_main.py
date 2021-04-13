#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import sys
import numpy as np

import auto_usefulness_dataProcess as dp
import auto_usefulness_model as model

debug = False

if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in auto_usefulness_main.py ...")

    # 1.读取评论文本内容+预处理
    print("》》》开始读取文本数据。。。")
    contents = dp.readContents(debug)
    X_contents, word_index, maxlen = dp.contentsPadding(contents)

    # 2.读取腾讯词向量
    print("》》》读取腾讯词向量。。。")
    if debug:
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
    else:
        embedding_matrix = model.load_word2vec(word_index)
    print("embedding_matrix's shape = ", embedding_matrix.shape)

    dict_length = min(150000, len(word_index) + 1)
    print("dict_length = ", dict_length)

    # 3.读取语言特征向量+标签
    print("》》》读取语言特征向量。。。")
    origin_data, X_language_feature, Y = dp.readFromCSV(debug)

    if len(X_contents) != len(X_language_feature) or len(X_contents) != len(Y):
        print("X_contents' length = ", len(X_contents))
        print("X_language_feature' length = ", len(X_language_feature))
        print("Y' length = ", len(Y))
        print("长度不一致！！！")
        sys.exit(-1)

    # 5.构建深度学习模型
    print("》》》构建深度学习模型。。。")
    batch_size = 256
    epochs = [4, 5, 6]
    language_feature_nodes = [256, 512]
    cnn_nodes = [32, 64, 128, 256, 512]
    model_name = "cnn"
    if model_name.startswith("cnn"):
        cnn_nodes = [32]
        current_model = model.createCNNModel(maxlen, dict_length, embedding_matrix)
        model.trainModel(model_name, current_model, X_language_feature, X_contents, Y, 3, batch_size, debug)
    elif model_name.startswith("fusion"):
        for epoch in epochs:
            for language_feature_node in language_feature_nodes:
                for cnn_node in cnn_nodes:
                    model_name += "_epoch_" + str(epoch) + "_language_" + str(language_feature_node) + "_cnn_" + str(cnn_node)
                    current_model = model.createFusionVectorsModel(464, maxlen, dict_length, embedding_matrix, language_feature_node, cnn_node)
                    # 6.训练模型
                    print("》》》训练深度学习模型。。。")
                    model.trainModel(model_name, current_model, X_language_feature, X_contents, Y, epoch, batch_size, debug)
    elif model_name.startswith("lstm"):
        dim = 64
        current_model = model.createLSTMModel(maxlen, dict_length, embedding_matrix, dim)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_language_feature, X_contents, Y, 3, batch_size, debug)
    elif model_name.startswith("gru"):
        dim = 64
        current_model = model.createGRUModel(maxlen, dict_length, embedding_matrix, dim)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_language_feature, X_contents, Y, 3, batch_size, debug)
    elif model_name.startswith("mlp"):
        current_model = model.createMLPModel(maxlen, dict_length, embedding_matrix, 256)
        print("》》》训练深度学习模型。。。")
        model.trainModel(model_name, current_model, X_language_feature, X_contents, Y, 3, batch_size, debug)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of auto_usefulness_main.py...")


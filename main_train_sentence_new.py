#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.utils import class_weight

import KMeansCluster as KMC
import fuzzySystem as fsys
import featureFusion_sentence as ff_s
import test
import dataProcess_sentence as dp_s
import fasttext

import time
from tensorflow.keras import models
# from tensorflow.keras.experimental import export_saved_model
# from tensorflow.keras.experimental import load_from_saved_model


# 情感词典相关变量
# word_index = {}  # 词语-序号字典,脏乱:0
# index_word = {}  # 序号-词语字典，0:脏乱
word_feature = {}  # 词语的特征,脏乱:[category, intensity, polar]

# 一些超参数的设置
dict_length = 150000  # 词典的长度，后期可以测试？？
embedding_dim = 128  # 词嵌入的维度，后期可以测试？？
embeddings = []  # 词嵌入，后期可以使用预训练词向量??

DEBUG = False

if __name__ == "__main__":
    print(">>>begin in main_train.py ...")
    print("start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 获取本体词汇库的情感向量
    print("》》》获取本体词汇库的情感向量。。。")
    # KMC.read_csv_open()
    word_feature = KMC.read_excel_pandas()



    print("》》》获取输入语料。。。")
    ids = [0, 1, 2, 3, 4]
    ids_two = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    ids_three = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    # for ids in ids_three:  # 针对三个数据集
    # for ids in ids_two:  # 针对两个数据集
    for id in ids:  # 针对一个数据集、四个数据集、五个数据集
        dict_length = 150000
        sampling = True
        # origin_data, y_cols, all_data, field = dp_s.initDataForOne(id, DEBUG)
        # origin_data, y_cols, all_data, field = dp_s.initDataForTwo(ids[0], ids[1], sampling, DEBUG)
        # origin_data, y_cols, all_data, field = dp_s.initDataForThree(ids[0], ids[1], ids[2], sampling)
        # origin_data, y_cols, all_data, field = dp_s.initDataForFour(id, sampling, DEBUG)
        origin_data, y_cols, all_data, field = dp_s.initDataForFive(sampling, DEBUG)
        # 训练集 验证集 测试集 分割比例，包括两个数：训练集比例、验证集比例
        ratios = [0.7, 0.29]
        # 获取停用词
        stoplist = dp_s.getStopList()
        # 获取输入语料的文本（去标点符号和停用词后）
        print("》》》获取输入语料的文本（去标点符号和停用词后）。。。")
        input_texts = dp_s.processDataToTexts(origin_data, stoplist)

        train_length = len(origin_data)
        print("train_length = ", train_length)

        print("origin_data.shape", origin_data.shape)

        # 确定单个输入语料的长度
        # 输入语料的长度，后期可以测试不同长度对结果的影响，参考hotelDataEmbedding.py的处理？？
        maxlen = dp_s.calculate_maxlen(input_texts)
        # maxlen = 52
        print("maxlen = ", maxlen)

        dealed_train, dealed_val, train, val, word_index = dp_s.processData3(origin_data, stoplist, dict_length, maxlen, ratios)

        if DEBUG:
            embedding_matrix = np.zeros((len(word_index) + 1, 300))
        else:
            embedding_matrix = ff_s.load_word2vec(word_index)
        print("embedding_matrix's shape = ", embedding_matrix.shape)

        dict_length = min(dict_length, len(word_index) + 1)
        print("dict_length = ", dict_length)

        epochs = [3]
        batch_sizes = [128]
        learning_rate = 0.001
        filters = [128, 256]
        window_sizes = [3, 4, 5, 6]
        dropouts = [0.3]
        full_connects = [128]
        balanceds = [True]

        # 自动跑模型
        for epoch in epochs:
            for batch_size in batch_sizes:
                for cnn_filter in filters:
                    for window_size in window_sizes:
                        for full_connect in full_connects:
                            for dropout in dropouts:
                                for balanced in balanceds:
                                    exp_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + \
                                                 "_cnnFilter_" + str(cnn_filter) + "_windowSize_" + str(window_size) + \
                                                 "_fullConnected_" + str(full_connect)
                                    print("exp_name = ", exp_name)
                                    model_name = "CNN"
                                    for i in range(3):
                                        if model_name == 'CNN':
                                            model_name = 'CNN_' + field
                                            model = ff_s.create_cnn_model(maxlen, dict_length, cnn_filter, embedding_matrix, window_size, dropout)
                                        elif model_name == 'LSTM':
                                            model_name = 'LSTM_' + field
                                            model = ff_s.create_lstm_model(maxlen, dict_length, embedding_matrix, dropout)
                                        elif model_name == "GRU":
                                            model_name = "GRU_" + field
                                            model = ff_s.create_gru_model(maxlen, dict_length, embedding_matrix, dropout)
                                        elif model_name == "CNNBiLSTM":
                                            model_name = 'CNNBiLSTM_' + field
                                            dim = 64
                                            model = ff_s.create_cnn_bilstm_model(maxlen, dict_length, cnn_filter, embedding_matrix, window_size, dropout, dim)
                                        elif model_name == "CNNBiGRU":
                                            model_name = 'CNNBiGRU_' + field
                                            dim = 128
                                            model = ff_s.create_cnn_bigru_model(maxlen, dict_length, cnn_filter, embedding_matrix, window_size, dropout, dim)
                                        else:
                                            pass
                                        if not model_name.startswith("fuzzy"):
                                            ff_s.train_all_model(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name)

print(">>>This is the end of main_train_sentence_new.py...")


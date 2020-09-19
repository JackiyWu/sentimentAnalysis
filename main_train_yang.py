#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 句子级情感分析

import numpy as np

from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.utils import class_weight

import KMeansCluster as KMC
import fuzzySystem as fsys
import featureFusion_yang as ff_y
import fuzzySentiment as fsent
import test
import dataProcess_yang as dp_y
import fasttext

import time


# 情感词典相关变量
# word_index = {}  # 词语-序号字典,脏乱:0
# index_word = {}  # 序号-词语字典，0:脏乱
word_feature = {}  # 词语的特征,脏乱:[category, intensity, polar]

# 聚类中心，统一使用KMeansCluster中的值
# clusters_centers = [[0.99347826, 0.98695652, 2.], [0.38799172, 1.06376812, 3.], [1.67957958, 1.10690691, 1.01501502]]

# 一些超参数的设置
dict_length = 150000  # 词典的长度，后期可以测试？？
embedding_dim = 128  # 词嵌入的维度，后期可以测试？？
embeddings = []  # 词嵌入，后期可以使用预训练词向量??
'''
dealed_train = []  # 输入语料，训练集
dealed_val = []  # 输入语料，验证集
dealed_test = []  # 输入语料，测试集
y_cols = []
'''


if __name__ == "__main__":
    print(">>>begin in main_train.py ...")
    print("start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 获取输入语料
    # col_names = ["dynamic", "inside", "safety", "self", "useful"]
    col_names = ["outside"]
    for col_name in col_names:
        dict_length = 150000  # 词典的长度，后期可以测试？？
        origin_data, y_cols = dp_y.initData4(col_name)
        # origin_data, y_cols = dp_s.initData2(1)
        # print("origin_data = ", origin_data)
        print("y_cols = ", y_cols)

        # 训练集 验证集 测试集 分割比例，包括两个数：训练集比例、验证集比例
        ratios = [0.7, 0.29]


        # 获取停用词
        stoplist = dp_y.getStopList()

        # 获取输入语料的文本（去标点符号和停用词后）
        input_texts = dp_y.processDataToTexts(origin_data, stoplist)
        # print("input_texts = ", input_texts)

        # 确定单个输入语料的长度
        # 输入语料的长度，后期可以测试不同长度对结果的影响，参考hotelDataEmbedding.py的处理？？
        maxlen = dp_y.calculate_maxlen(input_texts)
        # maxlen = 52
        print("maxlen = ", maxlen)
        # maxlen = dp_y.calculate_maxlen(input_texts_add)

        # 处理输入语料，生成训练集、验证集、测试集
        dealed_train, dealed_val, dealed_test, train, val, test, texts, word_index = dp_y.processData(origin_data, stoplist,
                                                                                                      dict_length, maxlen, ratios)

        print("dealed_train's shape = ", dealed_train.shape)

        # 根据预训练词向量生成embedding_matrix
        embedding_matrix = ff_y.load_word2vec(word_index)
        # embedding_matrix = np.zeros((len(word_index) + 1, 300))
        print("embedding_matrix's shape = ", embedding_matrix.shape)

        dict_length = min(dict_length, len(word_index) + 1)
        print("dict_length = ", dict_length)

        # dict_length = max(dict_length, len(fea_dict) + 1)

        # 生成模型-编译
        # 定义cnn的filter
        epochs = [10]
        # epochs = [200, 250, 300]
        # epochs = [5, 10, 20, 50, 100, 150, 200, 250, 300]
        batch_sizes = [128]
        # batch_sizes = [8, 16, 32, 128, 256]
        learning_rates = [0.001]
        # learning_rates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.0005, 0.0001]
        filters = [128]
        # filters = [64, 8, 32, 256, 512]
        window_sizes = [6]
        # window_sizes = [1, 2, 4, 5, 6, 7, 8]
        # dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8]
        dropouts = [0.5]
        balanceds = [True]

        # baseline
        # filter = 64
        # learning_rate = 0.001
        # window_size = 3
        # epoch = 10
        # dropout = 0.5
        # batch_size = 64

        # 自动跑模型
        for epoch in epochs:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for filter in filters:
                        for window_size in window_sizes:
                            for dropout in dropouts:
                                for balanced in balanceds:
                                    for i in range(5):
                                        print("i = ", i)
                                        if epoch == 10 and batch_size == 64 and learning_rate == 0.001 and filter == 64 and window_size == 3:
                                            if dropout not in (0.6, 0.7):
                                                continue

                                        name = "new_epoch_" + str(epoch) + "_batch_size_" + str(batch_size) + \
                                               "_learningRate_" + str(learning_rate) + "_filter_" + str(filter) + \
                                               "_window_size_" + str(window_size) + "_dropout_" + str(dropout) + \
                                               "_balanced_" + str(balanced)

                                        print("name = ", name)

                                        # fusion_model = ff_s.create_fusion_model(fuzzy_maxlen, maxlen, dict_length, filter,
                                        #                                         embedding_matrix, window_size, dropout)
                                        fusion_model = ff_y.create_cnn_model(maxlen, dict_length, filter, embedding_matrix, window_size, dropout)
                                        # fusion_model = ff_s.create_cnn_model(maxlen, dict_length, filters)
                                        # fusion_model = ff_s.fasttext_model(fea_dict, maxlen)
                                        # fusion_model = ff_s.create_lstm_model(maxlen, dict_length)
                                        # fusion_model = ff_s.create_lstm_model(maxlen, dict_length, embedding_matrix, dropout)
                                        # plot_model(fusion_model, 'modelsImage/Multi_input_model3.png')

                                        # fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

                                        # epoch
                                        # epoch = 10

                                        # 实验id
                                        # experiment_id = "macro_test"

                                        # 训练模型-cnn
                                        ff_y.train_cnn_model(fusion_model, train, val, dealed_train, dealed_test, dealed_val
                                                             , epoch, name, batch_size, learning_rate, y_cols)

                                        # 保存模型
                                        print(">>>保存模型ing")
                                        model_path = "models/"
                                        # fusion_model.save("experiment_28.h5")
    '''
    '''

    print(">>>This is the end of main_train.py...")


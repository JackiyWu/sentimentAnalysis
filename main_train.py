#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.utils import class_weight

import KMeansCluster as KMC
import fuzzySystem as fsys
import featureFusion as ff
import test
import dataProcess as dp
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
    print(">>>begin in main_train.py ...%Y-%m%d %H:%M%S", time.localtime())

    # 获取本体词汇库的情感向量
    word_feature = KMC.read_excel()
    # check数据
    # print("word_feature = ", word_feature)

    # 获取聚类中心
    clusters_centers = KMC.clusters_centers
    print("clusters_centers = ", clusters_centers)

    # 获取输入语料
    origin_data, y_cols = dp.initData()
    # print("origin_data = ", origin_data)
    print("y_cols = ", y_cols)

    # 统计输入语料中不同属性-不同标签的样本数
    sample_statistics = dp.calculate_sample_number(origin_data)
    print("sample_statistics = ", sample_statistics)

    # 把细粒度属性标签转为粗粒度属性标签
    # dp.processDataToTarget(origin_data)

    # 训练集 验证集 测试集 分割比例，包括两个数：训练集比例、验证集比例
    ratios = [0.6, 0.35]

    # 获取停用词
    stoplist = dp.getStopList()

    # 获取输入语料的文本（去标点符号和停用词后）
    input_texts = dp.processDataToTexts(origin_data, stoplist)
    # print("input_texts = ", input_texts)

    # 获取输入语料的情感向量特征
    input_word_feature = fsys.calculate_sentiment_words_feature(input_texts, word_feature)
    # print("input_word_feature = ", input_word_feature)

    # 根据情感向量特征计算得到情感隶属度特征
    dealed_train_fuzzy, dealed_val_fuzzy, dealed_test_fuzzy = fsys.cal_fuzzy_membership_degree(input_word_feature,
                                                                                               clusters_centers,
                                                                                               input_texts, ratios)
    # print("dealed_train_fuzzy = ", dealed_train_fuzzy)
    print("dealed_train_fuzzy's shape = ", dealed_train_fuzzy.shape)

    fuzzy_maxlen = fsys.calculate_input_dimension(dealed_train_fuzzy)
    print("fuzzy_maxlen = ", fuzzy_maxlen)

    # 获取增广特征
    # input_texts_add, fea_dict = fasttext.get_add_feature(input_texts)

    # 确定单个输入语料的长度
    # 输入语料的长度，后期可以测试不同长度对结果的影响，参考hotelDataEmbedding.py的处理？？
    maxlen = dp.calculate_maxlen(input_texts)
    print("maxlen = ", maxlen)
    # maxlen = dp.calculate_maxlen(input_texts_add)

    # 处理输入语料，生成训练集、验证集、测试集
    dealed_train, dealed_val, dealed_test, train, val, test, texts, word_index = dp.processData(origin_data, stoplist,
                                                                                    dict_length, maxlen, ratios)

    # fasttext
    # dealed_train, dealed_val, dealed_test, train, val, test = fasttext.processData(input_texts_add, origin_data,
    #                                                                                maxlen, ratios)
    # print("dealed_train = ", dealed_train)
    # print("train = ", train)
    # print("dealed_train = ", dealed_train)
    print("dealed_train's shape = ", dealed_train.shape)

    # 生成模型-编译
    # 定义cnn的filter
    filters = [64, 32]

    # 根据预训练词向量生成embedding_matrix
    # embedding_matrix = ff.load_word2vec(word_index)
    # print("embedding_matrix's shape = ", embedding_matrix.shape)

    dict_length = max(dict_length, len(word_index) + 1)
    print("dict_length = ", dict_length)

    # dict_length = max(dict_length, len(fea_dict) + 1)

    fusion_model = ff.create_fusion_model(fuzzy_maxlen, maxlen, dict_length, filters)
    # fusion_model = ff.create_fusion_model(fuzzy_maxlen, maxlen, dict_length, filters, embedding_matrix)
    # fusion_model = ff.create_cnn_model(maxlen, dict_length, filters, embedding_matrix)
    # fusion_model = ff.create_cnn_model(maxlen, dict_length, filters, embedding_matrix)
    # fusion_model = ff.fasttext_model(fea_dict, maxlen)
    # fusion_model = ff.create_lstm_model(maxlen, dict_length)
    # fusion_model = ff.create_lstm_model(maxlen, dict_length, embedding_matrix)
    plot_model(fusion_model, 'modelsImage/Multi_input_model4.png')

    # fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # 处理样本不平衡
    # class_weights = [{0: 1, 1: 13, 2: 15, 3: 1.5}, {0: 1.5, 1:4, 3: 3, 3: 1}, {0: 1.3, 1: 3.2, 2: 1.5, 3: 1},
    #                  {0: 1.3, 1: 5, 2: 4.8, 3: 1}, {0: 21, 1: 7, 2: 3, 3: 1}, {0: 49, 1: 7, 2: 3, 1:1}]
    # class_weights = {0: 1, 1: 1, 2: 1, 3: 1}
    class_weights = dp.calculate_class_weight(sample_statistics)

    # epoch
    epoch = 10

    # 训练模型
    ff.train_model(fusion_model, train, val, dealed_train_fuzzy, dealed_train, dealed_test_fuzzy, dealed_test,
                   dealed_val_fuzzy, dealed_val, y_cols, class_weights, epoch)

    # 训练模型-cnn
    # ff.train_cnn_model(fusion_model, train, val, dealed_train, dealed_test, dealed_val, y_cols, class_weights, epoch)
    # ff.train_fasttext_model(fusion_model, train, val, dealed_train, dealed_test, dealed_val, y_cols, class_weights, epoch)

    print(">>>This is the end of main_train.py...")


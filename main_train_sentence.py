#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 句子级情感分析

import numpy as np

from keras.utils.np_utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.utils import class_weight

import sentimentAnalysis.KMeansCluster as KMC
import sentimentAnalysis.fuzzySystem as fsys
import sentimentAnalysis.featureFusion_sentence as ff_s
import fuzzySentiment as fsent
import test
import sentimentAnalysis.dataProcess_sentence as dp_s
import fasttext

import time
from tensorflow.keras import models
from tensorflow.keras.experimental import export_saved_model
from tensorflow.keras.experimental import load_from_saved_model


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

    # 获取本体词汇库的情感向量
    word_feature = KMC.read_excel2()
    # word_feature = KMC.read_excel()
    # check数据
    # print("word_feature = ", word_feature)

    # 获取聚类中心
    clusters_centers = KMC.clusters_centers
    print("clusters_centers = ", clusters_centers)

    # 获取输入语料
    origin_data, y_cols = dp_s.initData3()
    # origin_data, y_cols = dp_s.initData2(1)
    # print("origin_data = ", origin_data)
    print("y_cols = ", y_cols)

    # 训练集 验证集 测试集 分割比例，包括两个数：训练集比例、验证集比例
    ratios = [0.7, 0.25]

    # 获取停用词
    stoplist = dp_s.getStopList()

    # 获取输入语料的文本（去标点符号和停用词后）
    input_texts = dp_s.processDataToTexts(origin_data, stoplist)
    # print("input_texts = ", input_texts)

    # 获取输入语料的情感向量特征
    input_word_feature = fsys.calculate_sentiment_words_feature(input_texts, word_feature)
    # print("input_word_feature = ", input_word_feature)

    # 根据情感向量特征计算得到情感隶属度特征
    # dealed_train_fuzzy, dealed_val_fuzzy, dealed_test_fuzzy = fsys.cal_fuzzy_membership_degree(input_word_feature,
    #                                                                                            clusters_centers,
    #                                                                                            input_texts, ratios)

    # 根据词汇本体库的情感向量特征计算得到三类情感特征值
    dealed_train_fuzzy, dealed_val_fuzzy, dealed_test_fuzzy = fsys.calculate_fuzzy_feature(input_word_feature, ratios)
    # print("dealed_train_fuzzy", dealed_train_fuzzy)
    # print("dealed_val_fuzzy", dealed_val_fuzzy)
    # print("dealed_test_fuzzy", dealed_test_fuzzy)

    # print("dealed_train_fuzzy = ", dealed_train_fuzzy)
    print("dealed_train_fuzzy's shape = ", dealed_train_fuzzy.shape)

    fuzzy_maxlen = fsys.calculate_input_dimension(dealed_train_fuzzy)
    print("fuzzy_maxlen = ", fuzzy_maxlen)

    # 获取增广特征
    # input_texts_add, fea_dict = fasttext.get_add_feature(input_texts)

    # 确定单个输入语料的长度
    # 输入语料的长度，后期可以测试不同长度对结果的影响，参考hotelDataEmbedding.py的处理？？
    maxlen = dp_s.calculate_maxlen(input_texts)
    # maxlen = 52
    print("maxlen = ", maxlen)
    # maxlen = dp_s.calculate_maxlen(input_texts_add)

    # 处理输入语料，生成训练集、验证集、测试集
    dealed_train, dealed_val, dealed_test, train, val, test, texts, word_index = dp_s.processData(origin_data, stoplist,
                                                                                                  dict_length, maxlen, ratios)

    # fasttext
    # dealed_train, dealed_val, dealed_test, train, val, test = fasttext.processData(input_texts_add, origin_data,
    #                                                                                maxlen, ratios)
    # print("dealed_train = ", dealed_train)
    # print("train = ", train)
    # print("dealed_train = ", dealed_train)
    print("dealed_train's shape = ", dealed_train.shape)

    # 根据预训练词向量生成embedding_matrix
    embedding_matrix = ff_s.load_word2vec(word_index)
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
    dropouts = [0.6]
    full_connecteds = [256]
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
                                for full_connected in full_connecteds:
                                    for i in range(1):
                                        print("i = ", i)
                                        if epoch == 10 and batch_size == 64 and learning_rate == 0.001 and filter == 64 and window_size == 3:
                                            if dropout not in (0.6, 0.7):
                                                continue
                                        name = "new_epoch_" + str(epoch) + "_batch_size_" + str(batch_size) + \
                                               "_learningRate_" + str(learning_rate) + "_filter_" + str(filter) + \
                                               "_window_size_" + str(window_size) + "_dropout_" + str(dropout) + \
                                               "_balanced_" + str(balanced) + "_full_connected_" + str(full_connected)
                                        print("name = ", name)

                                        fusion_model = ff_s.create_fusion_model(fuzzy_maxlen, maxlen, dict_length,
                                                                                filter, embedding_matrix, window_size,
                                                                                dropout, full_connected)
                                        # fusion_model = ff_s.create_cnn_model(maxlen, dict_length, filter, embedding_matrix, window_size, dropout)
                                        # fusion_model = ff_s.create_cnn_model(maxlen, dict_length, filters)
                                        # fusion_model = ff_s.fasttext_model(fea_dict, maxlen)
                                        # fusion_model = ff_s.create_lstm_model(maxlen, dict_length)
                                        # fusion_model = ff_s.create_lstm_model(maxlen, dict_length, embedding_matrix, dropout)
                                        plot_model(fusion_model, 'modelsImage/Multi_input_model3.png')

                                        # fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

                                        # epoch
                                        # epoch = 10

                                        # 实验id
                                        # experiment_id = "macro_test"

                                        # 训练模型
                                        ff_s.train_model(fusion_model, train, val, dealed_train_fuzzy, dealed_train, dealed_test_fuzzy, dealed_test,
                                                       dealed_val_fuzzy, dealed_val, y_cols, epoch, name, batch_size, learning_rate, balanced)
                                                       # dealed_val_fuzzy, dealed_val, y_cols, class_weights)

                                        # 训练模型-cnn
                                        # ff_s.train_cnn_model(fusion_model, train, val, dealed_train, dealed_test, dealed_val, epoch,
                                        #                      name, batch_size, learning_rate)
                                        # ff_s.train_fasttext_model(fusion_model, train, val, dealed_train, dealed_test, dealed_val, epoch)

                                        # 保存模型
                                        print(">>>保存模型ing")
                                        model_path = "export_saved_model.h5"
                                        # fusion_model.save("test_new_model.h5")
                                        export_saved_model(fusion_model, model_path)

                                        # 读取模型
                                        # model = models.load_model("test_export_saved_model.h5")
                                        model = load_from_saved_model(model_path)

                                        # 读取模型并预测
                                        ff_s.load_predict(model, dealed_test_fuzzy, dealed_test, test["review"])

    print(">>>This is the end of main_train.py...")


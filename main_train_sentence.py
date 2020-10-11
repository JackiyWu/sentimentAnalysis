#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
***************************************************************↓↓↓↓↓↓↓↓句子级情感分析流程↓↓↓↓↓↓↓↓***************************************************************
    1.读取原始训练数据集origin_data,去掉换行符、空格
    1.1 统计一下输入语料的长度
    1.2 对评论语句进行补齐[pad]

    2.加载bert模型bert_model
    3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings
    3.1 保存字符向量character_embeddings、句子级向量sentence_embeddings至文件 character_embeddings.csv和sentence_embeddings.csv,格式
    3.2 直接读取两个向量文件

    4.对sentence_embeddings进行聚类，得到三个聚类中心cluster_centers，并输出到文件
    5.计算每条评论的特征向量（字符级向量）到聚类中心的距离distance_from_feature_to_cluster
    6.使用cosin距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）

    7.将review_sentiment_membership_degree拼接在wcharacter_embeddings后面生成最终的词向量final_word_embeddings

    8.构建CNN模型

    9.训练模型
    9.1 数据集按照两种方式来训练：①直接划分比例，训练集、验证集、测试集按照 7:2:1划分；②交叉验证XXXX之后再写
    9.2 测试集的预测这里可能是有问题的，现在是每训练完一个属性就预测并打印，而不是整个模型训练完了之后才打印
    ***************************************************************↑↑↑↑↑↑↑↑句子级情感分析流程↑↑↑↑↑↑↑↑***************************************************************

'''

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

DEBUG = False

if __name__ == "__main__":
    print(">>>begin in main_train.py ...")
    print("start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 获取本体词汇库的情感向量
    print("》》》获取本体词汇库的情感向量。。。")
    # KMC.read_csv_open()
    word_feature = KMC.read_excel_pandas()
    # word_feature = KMC.read_excel2()
    # word_feature = KMC.read_excel()
    # check数据
    # print("word_feature = ", word_feature)

    # 获取聚类中心
    # clusters_centers = KMC.clusters_centers
    # print("clusters_centers = ", clusters_centers)

    # 获取输入语料
    print("》》》获取输入语料。。。")
    origin_data, y_cols = dp_s.initData3(DEBUG)
    # origin_data, y_cols = dp_s.initData2(1)
    # print("origin_data = ", origin_data)
    print("y_cols = ", y_cols)

    # 训练集 验证集 测试集 分割比例，包括两个数：训练集比例、验证集比例
    ratios = [0.7, 0.25]

    # 获取停用词
    stoplist = dp_s.getStopList()

    # 获取输入语料的文本（去标点符号和停用词后）
    print("》》》获取输入语料的文本（去标点符号和停用词后）。。。")
    input_texts = dp_s.processDataToTexts(origin_data, stoplist)
    # print("input_texts = ", input_texts)

    # 获取输入语料的情感向量特征
    # input_word_feature = fsys.calculate_sentiment_words_feature(input_texts, word_feature)
    # print("input_word_feature = ", input_word_feature)

    input_word_score = fsys.calculate_sentiment_score(input_texts, word_feature)
    # print("input_word_score = ", input_word_score)

    # 根据情感向量特征计算得到情感隶属度特征
    # dealed_train_fuzzy, dealed_val_fuzzy, dealed_test_fuzzy = fsys.cal_fuzzy_membership_degree(input_word_feature,
    #                                                                                            clusters_centers,
    #                                                                                            input_texts, ratios)

    # 根据情感得分计算三种极性的隶属度
    dealed_train_fuzzy, dealed_val_fuzzy, dealed_test_fuzzy = fsys.calculate_membership_degree_by_score(input_word_score, ratios)

    # 根据词汇本体库的情感向量特征计算得到三类情感特征值
    # dealed_train_fuzzy, dealed_val_fuzzy, dealed_test_fuzzy = fsys.calculate_fuzzy_feature(final_sentiment_feature, ratios)
    print("dealed_train_fuzzy.shape", dealed_train_fuzzy.shape)
    print("dealed_val_fuzzy.shape", dealed_val_fuzzy.shape)
    print("dealed_test_fuzzy.shape", dealed_test_fuzzy.shape)
    '''
    '''

    # print("dealed_train_fuzzy = ", dealed_train_fuzzy)
    # print("dealed_train_fuzzy's shape = ", dealed_train_fuzzy.shape)

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
    epochs = [5]
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
    full_connecteds = [128, 256]
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
                                    for i in range(20):
                                        print("i = ", i)
                                        # if epoch == 10 and batch_size == 64 and learning_rate == 0.001 and filter == 64 and window_size == 3:
                                        #     if dropout not in (0.6, 0.7):
                                        #         continue
                                        name = "new_epoch_" + str(epoch) + "_batch_size_" + str(batch_size) + \
                                               "_learningRate_" + str(learning_rate) + "_filter_" + str(filter) + \
                                               "_window_size_" + str(window_size) + "_dropout_" + str(dropout) + \
                                               "_balanced_" + str(balanced) + "_full_connected_" + str(full_connected)
                                        print("name = ", name)

                                        model_name = "cnn_financial"
                                        if model_name.startswith("fusion"):
                                            fusion_model = ff_s.create_fusion_model(fuzzy_maxlen, maxlen, dict_length,
                                                                                    filter, embedding_matrix, window_size,
                                                                                    dropout, full_connected)
                                            ff_s.train_model(fusion_model, train, val, dealed_train_fuzzy, dealed_train, dealed_test_fuzzy, dealed_test,
                                                           dealed_val_fuzzy, dealed_val, y_cols, epoch, name, batch_size, learning_rate, balanced, model_name)
                                        elif model_name.startswith("cnn"):
                                            fusion_model = ff_s.create_cnn_model(maxlen, dict_length, filter, embedding_matrix, window_size, dropout)
                                            ff_s.train_cnn_model(fusion_model, train, val, dealed_train, dealed_test, dealed_val, epoch,
                                                                 name, batch_size, learning_rate, model_name)
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

                                        # 训练模型
                                        #                dealed_val_fuzzy, dealed_val, y_cols, class_weights)

                                        # 训练模型-cnn
                                        # ff_s.train_fasttext_model(fusion_model, train, val, dealed_train, dealed_test, dealed_val, epoch)

                                        # 保存模型
                                        print(">>>保存模型ing")
                                        # model_path = "export_saved_model.h5"
                                        # fusion_model.save("test_new_model.h5")
                                        # export_saved_model(fusion_model, model_path)

                                        # 读取模型
                                        # model = models.load_model("test_export_saved_model.h5")
                                        # model = load_from_saved_model(model_path)

                                        # 读取模型并预测
                                        # ff_s.load_predict(model, dealed_test_fuzzy, dealed_test, test["review"])
    '''
    '''

    print(">>>This is the end of main_train.py...")


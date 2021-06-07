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

DEBUG = False

if __name__ == "__main__":
    print(">>>begin in main_train.py ...")
    print("start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # 获取本体词汇库的情感向量
    print("》》》获取本体词汇库的情感向量。。。")
    # KMC.read_csv_open()
    word_feature = KMC.read_excel_pandas()

    print("》》》获取输入语料。。。")
    ids = [3]
    # ids = [0, 1, 2, 3, 4]
    ids_two = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    ids_three = [[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4], [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
    ids_five = [0]
    # for ids in ids_two:  # 针对两个数据集
    # for ids in ids_three:  # 针对三个数据集
    # for id in ids_five:
    for id in ids:  # 针对一个数据集、四个数据集
        dict_length = 150000
        sampling = False
        # origin_data, y_cols, all_data, field, data_tests, data_test_names = dp_s.initDataForOne(id, DEBUG)
        # origin_data, y_cols, all_data, field, data_tests, data_test_names = dp_s.initDataForTwo(ids[0], ids[1], sampling, DEBUG)
        # origin_data, y_cols, all_data, field, data_tests, data_test_names = dp_s.initDataForThree(ids[0], ids[1], ids[2], sampling, DEBUG)
        origin_data, y_cols, all_data, field, data_tests, data_test_names = dp_s.initDataForFour(id, sampling, DEBUG)
        # origin_data, y_cols, all_data, field, data_tests, data_test_names = dp_s.initDataForFive(sampling, DEBUG)
        # print("origin_data's length = ", len(origin_data))
        # 训练集 验证集 测试集 分割比例，包括两个数：训练集比例、验证集比例
        ratio = 0.9
        # 获取停用词
        stoplist = dp_s.getStopList()
        # 获取输入语料的文本（去标点符号和停用词后）
        print("》》》获取输入语料的文本（去标点符号和停用词后）。。。")
        input_texts = dp_s.processDataToTexts(origin_data, stoplist)
        print("input_texts's length = ", len(input_texts))
        # 获取测试集的输入语料文本
        input_texts_tests = []
        for t in data_tests:
            input_texts_tests.append(dp_s.processDataToTexts(t, stoplist))

        # 确定单个输入语料的长度
        # 输入语料的长度，后期可以测试不同长度对结果的影响，参考hotelDataEmbedding.py的处理？？
        # print(input_texts)
        # maxlen = dp_s.calculate_maxlen(input_texts)
        # 后期测试
        # maxlens = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]
        maxlen = 150
        # for maxlen in maxlens:
        print("maxlen = ", maxlen)

        # 获取模糊特征
        input_word_score = fsys.calculate_sentiment_score(input_texts, word_feature)
        input_word_score_tests = []
        for text in input_texts_tests:
            input_word_score_tests.append(fsys.calculate_sentiment_score(text, word_feature))
        # 根据情感得分计算三种极性的隶属度
        dealed_fuzzy = fsys.calculate_membership_degree_by_score_no_split(input_word_score, maxlen)
        dealed_fuzzy_tests = []
        for score in input_word_score_tests:
            dealed_fuzzy_tests.append(fsys.calculate_membership_degree_by_score_no_split(score, maxlen))
        dealed_fuzzy_length = len(dealed_fuzzy)
        print("dealed_fuzzy_length = ", dealed_fuzzy_length)
        fuzzy_maxlen = fsys.calculate_input_dimension(dealed_fuzzy)
        print("fuzzy_maxlen = ", fuzzy_maxlen)

        train_length = len(origin_data)
        print("train_length = ", train_length)

        print("origin_data.shape", origin_data.shape)

        dealed_train, dealed_val, train, val, word_index, dealed_data, tokenizer, dealed_fuzzy_train, dealed_fuzzy_val = dp_s.processData3(origin_data, stoplist, dict_length, maxlen, ratio, dealed_fuzzy)
        print("dealed_data's type = ", type(dealed_data))
        # 生成测试集的padding
        dealed_tests = dp_s.generatePadding(tokenizer, stoplist, maxlen, data_tests)

        if DEBUG:
            embedding_matrix = np.zeros((len(word_index) + 1, 300))
        else:
            embedding_matrix = ff_s.load_word2vec(word_index)
        print("embedding_matrix's shape = ", embedding_matrix.shape)

        dict_length = min(dict_length, len(word_index) + 1)
        print("dict_length = ", dict_length)

        epoch = 20
        batch_size = 128
        # batch_sizes = [128, 256, 512]
        learning_rate = 0.001
        filters = [32]
        # filters = [8, 16, 32, 64, 128, 256, 512, 1024]
        # window_sizes = [1, 2, 3, 4, 5, 6, 7]
        window_sizes = [3]
        full_connects = [303]
        # full_connects = [16, 32, 64, 128, 256, 303, 512]
        lstm_dim1s = [64]
        gru_dims = [128]
        # gru_dims = [16, 32, 64, 128, 256]
        times = 10
        mlp_connects = [128]
        metrics_name = "mae"
        model_names = ['Fusion']
        # model_names = ['FusionLSTM', 'Fusion', 'FusionMLP', 'FusionGRU']
        print("测试micro weighted macro时模型效果", model_names)
        # print("测试cnn_filter=8时模型效果", model_names)
        # print("测试Trainable为FALSE时模型效果", model_names)

        # 自动跑模型
        for model_name in model_names:
                # model_name = "FusionGRU"

            if model_name == 'CNN':
                for cnn_filter in filters:
                    for window_size in window_sizes:
                        exp_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_cnnFilter_" + str(cnn_filter) + "_windowSize_" + str(window_size)
                        print("exp_name = ", exp_name)
                        for i in range(times):
                            model = ff_s.create_cnn_model(maxlen, dict_length, cnn_filter, embedding_matrix, window_size, metrics_name)
                            y_val_pred_cnn = ff_s.train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names)
            elif model_name == 'textCNN':
                for cnn_filter in filters:
                    model_name = metrics_name + '_textCNN_' + field + '_maxlen_' + str(maxlen)
                    exp_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_cnnFilter_" + str(cnn_filter) + "_fullConnected_" + str(full_connect)
                    print("exp_name = ", exp_name)
                    for i in range(times):
                        model = ff_s.create_textCNN_model(maxlen, dict_length, cnn_filter, embedding_matrix, metrics_name)
                        ff_s.train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names)
            elif model_name == 'LSTM':
                for dim1 in lstm_dim1s:
                    model_name = 'LSTM_' + field + "_maxlen_" + str(maxlen)
                    exp_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_dim1_" + str(dim1) + "_fullConnected_" + str(full_connect)
                    print("exp_name = ", exp_name)
                    # for i in range(times):
                    model = ff_s.create_lstm_model(maxlen, dict_length, embedding_matrix, dim1)
                    y_val_pred = ff_s.train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names)
            elif model_name == "GRU":
                for dim in gru_dims:
                    model_name = "GRU_" + field + "_maxlen_" + str(maxlen) + "_sampling_" + str(sampling)
                    exp_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + \
                               "_dim_" + str(dim) + "_fullConnected_" + str(full_connect)
                    print("exp_name = ", exp_name)
                    for i in range(times):
                        model = ff_s.create_gru_model(maxlen, dict_length, embedding_matrix, dim)
                        ff_s.train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names)
            elif model_name == "CNNBiLSTM":
                model_name = 'CNNBiLSTM_' + field + "_maxlen_" + str(maxlen)
                exp_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_fullConnected_" + str(full_connect)
                model = ff_s.create_cnn_bilstm_model(maxlen, dict_length, embedding_matrix)
                ff_s.train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names)
            elif model_name == "CNNBiGRU":
                for cnn_filter in filters:
                    for window_size in window_sizes:
                        for dim in gru_dims:
                            model_name = 'CNNBiGRU_' + field + "_maxlen_" + str(maxlen)
                            for i in range(3):
                                model = ff_s.create_cnn_bigru_model(maxlen, dict_length, cnn_filter, embedding_matrix, window_size, dim)
                                ff_s.train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names)
            elif model_name == "MLP":
                model_name = "MLP_" + field + "_maxlen_" + str(maxlen)
                exp_name = "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_fullConnected_" + str(128)
                # for i in range(times):
                model = ff_s.create_mlp_model(maxlen, dict_length, embedding_matrix)
                y_val_pred = ff_s.train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names)
            elif model_name == "lstm_attention":
                for dim in lstm_dim1s:
                    model_name = "embedding1_lstm_attention"
                    for i in range(times):
                        exp_name = field + "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_dim_" + str(dim)
                        model = ff_s.create_fusion_attention_lstm_model(maxlen, dict_length, embedding_matrix, dim)
                        y_val_pred_fusion_attention_lstm = ff_s.train_fusion_model_no_cross(model, train, val, dealed_train, dealed_val, dealed_fuzzy_train, dealed_fuzzy_val, epoch,
                                                                                            exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names, dealed_fuzzy_tests)
            elif model_name == "Fusion":
                for full_connect in full_connects:
                    for cnn_filter in filters:
                        for window_size in window_sizes:
                            for i in range(times):
                                model_name = "Fusion_micro_weighted_macro_sampling_" + str(sampling)
                                exp_name = field + "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_cnn_filter_" + str(cnn_filter) + "_window_size_" + str(window_size)
                                # exp_name = field + "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_cnn_filter_" + str(cnn_filter) + "_window_size_" + str(window_size) + "_fullConnected_" + str(full_connect)
                                model = ff_s.create_fusion_model_new(maxlen, dict_length, cnn_filter, embedding_matrix, window_size, full_connect)
                                y_val_pred_fusion_cnn = ff_s.train_fusion_model_no_cross(model, train, val, dealed_train, dealed_val, dealed_fuzzy_train, dealed_fuzzy_val, epoch,
                                                                 exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names, dealed_fuzzy_tests)
            elif model_name == "FusionLSTM":
                for dim in lstm_dim1s:
                    model_name = "FusionLSTM"
                    exp_name = field + "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_dim_" + str(dim)
                    model = ff_s.create_fusion_model_lstm(maxlen, dict_length, embedding_matrix, dim)
                    y_val_pred_fusion = ff_s.train_fusion_model_no_cross(model, train, val, dealed_train, dealed_val, dealed_fuzzy_train, dealed_fuzzy_val, epoch,
                                                                        exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names, dealed_fuzzy_tests)
            elif model_name == "FusionGRU":
                model_name = "FusionGRU"
                exp_name = field + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_fullConnected_" + str(128)
                for dim in gru_dims:
                    model = ff_s.create_fusion_model_gru(maxlen, dict_length, embedding_matrix, dim)
                    ff_s.train_fusion_model_no_cross(model, train, val, dealed_train, dealed_val, dealed_fuzzy_train, dealed_fuzzy_val, epoch,
                                                     exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names, dealed_fuzzy_tests)
            elif model_name == "FusionMLP":
                for mlp_node in mlp_connects:
                    model_name = "FusionMLP"
                    exp_name = field + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_node_" + str(mlp_node)
                    for i in range(times):
                        model = ff_s.create_fusion_model_mlp(maxlen, dict_length, embedding_matrix, mlp_node)
                        ff_s.train_fusion_model_no_cross(model, train, val, dealed_train, dealed_val, dealed_fuzzy_train, dealed_fuzzy_val, epoch,
                                                         exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names, dealed_fuzzy_tests)
            elif model_name == "Fusion_attention_lstm":
                for dim in lstm_dim1s:
                    model_name = "Fusion_attention_lstm"
                    for i in range(times):
                        exp_name = field + "epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_dim_" + str(dim)
                        model = ff_s.create_fusion_attention_lstm_model(maxlen, dict_length, embedding_matrix, dim)
                        y_val_pred_fusion_attention_lstm = ff_s.train_fusion_model_no_cross(model, train, val, dealed_train, dealed_val, dealed_fuzzy_train, dealed_fuzzy_val, epoch,
                                                                             exp_name, batch_size, model_name, dealed_tests, data_tests, data_test_names, dealed_fuzzy_tests)
            else:
                pass
            print("训练结束。。")
        # 保存验证集的预测结果
        '''
        if len(model_names) == 2:
            print("y_val_pred_cnn's type = ", type(y_val_pred_cnn))
            print("y_val_pred_fusion_cnn's type = ", type(y_val_pred_fusion_cnn))
            ff_s.save_validation_result_to_csv(val, dealed_fuzzy_val, y_val_pred_cnn.tolist(), y_val_pred_fusion_cnn.tolist())
        '''

print(">>>This is the end of main_train_sentence_new.py...")


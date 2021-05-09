#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import codecs
import csv
import math
from itertools import chain
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import keras
from keras.layers import Input, Flatten, Dense, Dropout, Activation, GRU, Bidirectional, Conv1D, LSTM
from keras.layers import Embedding, merge, Lambda, Reshape, BatchNormalization, MaxPool1D, GlobalAveragePooling1D
from keras import Model, Sequential
from keras.utils import to_categorical
from keras import regularizers
from keras.layers.merge import concatenate
from keras.optimizers import Adam

from tensorflow.keras.layers import SeparableConvolution1D
import tensorflow as tf

from keras_bert import Tokenizer, load_trained_model_from_checkpoint, AdamWarmup, calc_train_steps

import auto_absa_config as config
import auto_absa_dataProcess as dp


# Bert+CNN模型
# 不提取词向量，直接用bert连接后面的模型
def createBertCNNModel(filter, window_size):
    print("开始构建Bert+CNN模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path, trainable=True)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])
        cnn = Conv1D(filter, window_size, name='conv')(x)
        # cnn = BatchNormalization()(cnn)
        cnn = MaxPool1D(name='max_pool')(cnn)

        flatten = Flatten()(cnn)

        x = Dense(32, activation='relu', name='dense_1')(flatten)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model([x1_in, x2_in], p)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])

    model.summary()
    print(">>>Bert+CNN模型构建结束。。。")

    return model


# Bert+GRU模型
# 不提取词向量，直接用bert连接后面的模型
def createBertGRUModel(dim):
    print("开始构建BertGRU模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path, trainable=True)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        bi_gru = Bidirectional(GRU(dim, name="gru_1"))(x)

        x = Dense(64, activation='relu', name='dense_1')(bi_gru)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    print("BertGRU模型构建结束。。。")

    return model


# Bert+单层CNN+BiGRU模型
# 不提取词向量，直接用bert连接后面的模型
def createBertCNNBiGRUModel(cnn_filter, cnn_window_size, gru_output_dim, debug=False):
    print("开始构建CNNBiGRU模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path, trainable=True)

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        cnn = Conv1D(cnn_filter, cnn_window_size, padding='same', strides=1, activation='relu', name='conv')(x)
        cnn = MaxPool1D(name='max_pool')(cnn)

        bi_gru = Bidirectional(GRU(gru_output_dim, name="gru_1"))(cnn)

        x = Dense(64, activation='relu', name='dense_1')(bi_gru)
        # x = Dropout(0.4, name='dropout')(x)
        x = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=x)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    print("BertCNNBiGRU模型构建完成。。。")

    return model


# Bert+LSTM模型
# 不提取词向量，直接用bert连接后面的模型
def createBertLSTMModel(dim):
    print("开始构建BertLSTM模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        bert_model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path, trainable=True)
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])

        lstm = LSTM(dim, return_sequences=False, name='lstm1')(x)

        x = Dense(64, activation='relu', name='dense_1')(lstm)
        x = Dropout(0.4, name='dropout')(x)
        p = Dense(4, activation='softmax', name='softmax')(x)

        model = Model(inputs=[x1_in, x2_in], outputs=p)

        model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    print("BertLSTM模型构建结束。。。")

    return model


# 训练bert模型
def trainBert(experiment_name, model, X, Y, y_cols_name, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, debug=False):
    print("勿扰！训练模型ing。。。in trainBert。。。model_name = ", model_name)

    length = len(Y)
    length_validation = len(Y_validation)
    print(">>>y's length = ", length)

    F1_scores = 0

    for index, col in enumerate(y_cols_name):
        print("Current col is: ", col)
        origin_data_current_col = Y[col]
        origin_data_current_col = list(origin_data_current_col)
        # origin_data_current_col = np.array(origin_data_current_col)

        # print("y_val = ", y_val)
        origin_data_current_col_val = Y_validation[col]
        origin_data_current_col_val = list(origin_data_current_col_val)
        # origin_data_current_col_val = np.array(origin_data_current_col_val)
        # print(y_val)

        history = model.fit(dp.generateSetForBert(X, origin_data_current_col, batch_size, tokenizer), steps_per_epoch=math.ceil(length / (batch_size)),
                            epochs=epoch, batch_size=batch_size, verbose=1, validation_steps=math.ceil(length_validation / (batch_size_validation)),
                            validation_data=dp.generateSetForBert(X_validation, origin_data_current_col_val, batch_size_validation, tokenizer))
        # 预测验证集
        y_val_pred = model.predict(dp.generateXSetForBert(X_validation, length_validation, batch_size_validation, tokenizer), steps=math.ceil(length_validation / batch_size_validation))

        # 预测专家评论数据
        print("》》》正在预测专家评论数据。。。")
        X_expert = dp.readExpertData()
        length_expert = len(X_expert)
        print("length_expert = ", length_expert)
        y_expert_pred = model.predict(dp.generateXSetForBert(X_expert, length_expert, batch_size_validation, tokenizer), steps=math.ceil(length_expert / batch_size_validation))
        # 将预测结果存入文件
        print("before y_expert_pred's length = ", len(y_expert_pred))
        y_expert_pred = np.argmax(y_expert_pred, axis=1).tolist()
        print("after y_expert_pred's length = ", len(y_expert_pred))
        save_expert_to_csv(y_expert_pred, debug)

        print("y_val_pred's length = ", len(y_val_pred))
        print("y_validation's length = ", length_validation)

        # print("y_val_pred: ", y_val_pred)
        # 将验证集的预测结果存入文件
        # save_predict_result_to_csv(y_val_pred, col)

        y_val_pred = np.argmax(y_val_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        print("y_val[20] = ", list(origin_data_current_col_val)[:20])
        print("y_val_pred[20] = ", list(y_val_pred)[:20])
        precision, recall, fscore, support = score(origin_data_current_col_val, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        report = classification_report(origin_data_current_col_val, y_val_pred, digits=4, output_dict=True)
        print(report)

        F1_score = f1_score(y_val_pred, origin_data_current_col_val, average='macro')
        F1_scores += F1_score
        print('第', index, '个属性', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, origin_data_current_col_val))
        print("%Y-%m%d %H:%M:%S", time.localtime())

        # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
        save_result_to_csv(report, F1_score, experiment_name, model_name, col, debug)

    print('all F1_score:', F1_scores / len(y_cols_name))

    print(">>>end of train_cnn_model function in featureFusion.py。。。")


# 将专家经验的预测结果写入文件
def save_expert_to_csv(y_expert_pred, debug):
    print("in save_expert_to_csv y_expert_pred's length = ", len(y_expert_pred))
    if debug:
        path = "result/result_yang/absa_expert_pred_debug.csv"
    else:
        path = "result/result_yang/absa_expert_pred.csv"
    # 写入csv文件
    with codecs.open(path, "a", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(y_expert_pred)
        f.close()


# 把结果保存到csv
# report是classification_report生成的字典结果
def save_result_to_csv(report, f1_score, experiment_id, model_name, col_name, debug):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [experiment_id, col_name, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, f1_score, accuracy]

    if debug:
        path = "result_yang/auto_absa_" + model_name + "_debug.csv"
    else:
        path = "result_yang/auto_absa_" + model_name + ".csv"
    with codecs.open(path, "a", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


# 加载tokenizer
def get_tokenizer():
    token_dict = {}
    with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = Tokenizer(token_dict)

    return tokenizer
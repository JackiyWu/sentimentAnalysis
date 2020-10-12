#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import codecs
import csv
import math

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

from tensorflow.keras.layers import SeparableConvolution1D

from keras_bert import Tokenizer, load_trained_model_from_checkpoint

import absa_config as config
import absa_dataProcess as dp

# 一些超参数
TOKEN_DICT = {}


# 创建bert模型
def createBertEmbeddingModel():
    with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            TOKEN_DICT[token] = len(TOKEN_DICT)

    model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path)

    return model


# CNN模型
def createCNNModel(maxlen, embedding_dim, debug=False):
    if debug:
        embedding_dim = 8
    print("开始构建CNN模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn = Conv1D(64, 4, padding='same', strides=1, activation='relu', name='conv')(tensor_input)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPool1D(name='max_pool')(cnn)

    flatten = Flatten()(cnn)

    x = Dense(64, activation='relu', name='dense_1')(flatten)
    x = Dropout(0.4, name='dropout')(x)
    x = Dense(4, activation='softmax', name='softmax')(x)

    model = Model(inputs=tensor_input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# GRU模型
def createGRUModel(maxlen, embedding_dim, dim_1, dim_2, debug=False):
    if debug:
        embedding_dim = 8
    print("开始构建GRU模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))

    bi_gru1 = Bidirectional(GRU(dim_1, activation='tanh', dropout=0.5, recurrent_dropout=0.4, return_sequences=True, name="gru_0"))(tensor_input)
    bi_gru1 = BatchNormalization()(bi_gru1)
    bi_gru2 = Bidirectional(GRU(dim_2, dropout=0.5, recurrent_dropout=0.5, name="gru_1"))(bi_gru1)
    bi_gru2 = BatchNormalization()(bi_gru2)

    flatten = Flatten()(bi_gru2)

    x = Dense(64, activation='relu', name='dense_1')(flatten)
    x = Dropout(0.4, name='dropout')(x)
    x = Dense(4, activation='softmax', name='softmax')(x)

    model = Model(inputs=tensor_input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# LSTM
def createLSTMModel(maxlen, embedding_dim, dim1, dim2, debug=False):
    if debug:
        embedding_dim = 8
    print("开始构建LSTM模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))

    lstm = Bidirectional(LSTM(dim1, return_sequences=True, name='lstm1'))(tensor_input)
    lstm = Bidirectional(LSTM(dim2, return_sequences=False, name='lstm2'))(lstm)

    x = Dense(64, activation='relu', name='dense_1')(lstm)
    x = Dropout(0.4, name='dropout')(x)
    x = Dense(4, activation='softmax', name='softmax')(x)

    model = Model(inputs=tensor_input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# 根据textCNN模型输出词向量
def createSeparableCNNModel(maxlen, embedding_dim, debug=False):
    if debug:
        embedding_dim = 8
    print(">>>开始构建SeparableCNN模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn1 = SeparableConvolution1D(200, 3, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_1")(tensor_input)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPool1D(pool_size=100)(cnn1)
    cnn2 = SeparableConvolution1D(200, 4, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_2")(tensor_input)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPool1D(pool_size=100)(cnn2)
    cnn3 = SeparableConvolution1D(200, 5, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_3")(tensor_input)
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = MaxPool1D(pool_size=100)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

    dropout = Dropout(0.2)(cnn)
    flatten = Flatten()(dropout)
    dense = Dense(512, activation='relu')(flatten)
    dense = BatchNormalization()(dense)
    dropout = Dropout(0.2)(dense)
    tensor_output = Dense(4, activation='softmax')(dropout)

    model = Model(inputs=tensor_input, outputs = tensor_output)
    # print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(">>>TextCNN模型构建结束。。。")
    return model


# 构建单层CNN+BiGRU
def createCNNBiGRUModel(maxlen, embedding_dim, cnn_filter, cnn_window_size, gru_output_dim_1, gru_output_dim_2, debug=False):
    if debug:
        embedding_dim = 8
    print("开始构建CNNBiGRU模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn = Conv1D(cnn_filter, cnn_window_size, padding='same', strides=1, activation='relu', name='conv')(tensor_input)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPool1D(name='max_pool')(cnn)

    dropout = Dropout(0.2)(cnn)
    # flatten = Flatten()(dropout)

    bi_gru1 = Bidirectional(GRU(gru_output_dim_1, activation='tanh', dropout=0.5, recurrent_dropout=0.4, return_sequences=True, name="gru_0"))(dropout)
    bi_gru1 = BatchNormalization()(bi_gru1)
    bi_gru2 = Bidirectional(GRU(gru_output_dim_2, dropout=0.5, recurrent_dropout=0.5, name="gru_1"))(bi_gru1)
    bi_gru2 = BatchNormalization()(bi_gru2)

    flatten = Flatten()(bi_gru2)

    x = Dense(64, activation='relu', name='dense_1')(flatten)
    x = Dropout(0.4, name='dropout')(x)
    x = Dense(4, activation='softmax', name='softmax')(x)

    model = Model(inputs=tensor_input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# 构建单层2CNN+BiGRU
def createMultiCNNBiGRUModel(maxlen, embedding_dim, cnn_filter, cnn_window_size_1, cnn_window_size_2, gru_output_dim_1, gru_output_dim_2, debug=False):
    if debug:
        embedding_dim = 8
    print("开始构建CNNBiGRU模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn1 = Conv1D(cnn_filter, cnn_window_size_1, padding='same', strides=1, activation='relu', name='conv')(tensor_input)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPool1D(name='max_pool')(cnn1)

    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn2 = Conv1D(cnn_filter, cnn_window_size_2, padding='same', strides=1, activation='relu', name='conv')(tensor_input)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPool1D(name='max_pool')(cnn2)

    cnn = concatenate([cnn1, cnn2], axis=-1)

    dropout = Dropout(0.2)(cnn)
    # flatten = Flatten()(dropout)

    bi_gru1 = Bidirectional(GRU(gru_output_dim_1, activation='tanh', dropout=0.5, recurrent_dropout=0.4, return_sequences=True, name="gru_0"))(dropout)
    bi_gru1 = BatchNormalization()(bi_gru1)
    bi_gru2 = Bidirectional(GRU(gru_output_dim_2, dropout=0.5, recurrent_dropout=0.5, name="gru_1"))(bi_gru1)
    bi_gru2 = BatchNormalization()(bi_gru2)

    flatten = Flatten()(bi_gru2)

    x = Dense(64, activation='relu', name='dense_1')(flatten)
    x = Dropout(0.4, name='dropout')(x)
    x = Dense(4, activation='softmax', name='softmax')(x)

    model = Model(inputs=tensor_input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model


# 构建模型CNN+BiGRU
def createSeparableCNNBiGRUModel(maxlen, embedding_dim, cnn_filter, window_size_1, window_size_2, window_size_3, debug=False):
    if debug:
        embedding_dim = 8
    print(">>>开始构建SeparableCNNBiGRU模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn1 = SeparableConvolution1D(cnn_filter, window_size_1, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_0")(tensor_input)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPool1D(pool_size=100)(cnn1)
    cnn2 = SeparableConvolution1D(cnn_filter, window_size_2, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_1")(tensor_input)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPool1D(pool_size=100)(cnn2)
    cnn3 = SeparableConvolution1D(cnn_filter, window_size_3, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_2")(tensor_input)
    cnn3 = BatchNormalization()(cnn3)
    cnn3 = MaxPool1D(pool_size=100)(cnn3)
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)

    dropout = Dropout(0.2)(cnn)
    # flatten = Flatten()(dropout)

    bi_gru1 = Bidirectional(GRU(128, activation='tanh', dropout=0.5, recurrent_dropout=0.4, return_sequences=True, name="gru_0"))(dropout)
    bi_gru1 = BatchNormalization()(bi_gru1)
    bi_gru2 = Bidirectional(GRU(256, dropout=0.5, recurrent_dropout=0.5, name="gru_1"))(bi_gru1)
    bi_gru2 = BatchNormalization()(bi_gru2)

    flatten = Flatten()(bi_gru2)

    dense = Dense(512, activation='relu')(flatten)
    dense = BatchNormalization()(dense)
    dropout = Dropout(0.2)(dense)
    tensor_output = Dense(4, activation='softmax')(dropout)

    model = Model(inputs=tensor_input, outputs=tensor_output)
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(">>>TextCNNBiGRUModel模型构建结束。。。")
    return model


# MLP
def createMLPModel(maxlen, embedding_dim, dense_dim, debug=False):
    if debug:
        embedding_dim = 8
    print(">>>开始构建MLP模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    flatten = Flatten()(tensor_input)
    dense = Dense(dense_dim, activation='relu')(flatten)
    dropout = Dropout(0.4)(dense)
    tensor_output = Dense(4, activation='softmax')(dropout)

    model = Model(inputs=tensor_input, outputs=tensor_output)
    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # print(">>>TextCNNBiGRUModel模型构建结束。。。")
    return model


# 训练模型，直接从文件中读取词向量
def trainModelFromFile(experiment_name, model, X_path, y, y_cols_name, X_val_path, y_val, model_name, epoch=3, batch_size=128, debug=False):
    print("勿扰！训练模型ing。。。in trainModelFromFile。。。")
    if len(X_path.strip()) > 0:
        print("从文件中直接读取词向量。。。")

    length = len(y)
    length_validation = len(y_val)
    print(">>>y's length = ", length)
    # if debug:
    #     y_cols_name = ["service"]
    batch_size_validation = batch_size

    F1_scores = 0
    F1_score = 0

    for index, col in enumerate(y_cols_name):
        # print("y = ", y)
        origin_data_current_col = y[col] + 2
        origin_data_current_col = np.array(origin_data_current_col)

        # print("y_val = ", y_val)
        origin_data_current_col_val = y_val[col] + 2
        origin_data_current_col_val = np.array(origin_data_current_col_val)
        # print(y_val)

        history = model.fit(dp.generateTrainSetFromFile(X_path, origin_data_current_col, batch_size, debug), steps_per_epoch=math.ceil(length / batch_size),
                            validation_data=dp.generateTrainSetFromFile(X_val_path, origin_data_current_col_val, batch_size, debug), validation_steps=math.ceil(length_validation / batch_size_validation),
                            batch_size=batch_size, epochs=epoch, verbose=2)

        # 预测验证集
        y_val_pred = model.predict(dp.generateXFromFile(X_val_path, length_validation, batch_size, debug), steps=math.ceil(length_validation / batch_size_validation))
        print("y_val_pred's length = ", len(y_val_pred))
        print("y_validation's length = ", length_validation)

        y_val_pred = np.argmax(y_val_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        print("y_val = ", list(origin_data_current_col_val))
        print("y_val_pred = ", list(y_val_pred))
        precision, recall, fscore, support = score(origin_data_current_col_val, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        report = classification_report(origin_data_current_col_val, y_val_pred, digits=4, output_dict=True)
        print(report)

        F1_score = f1_score(y_val_pred, origin_data_current_col_val, average='macro')
        F1_scores += F1_score
        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, origin_data_current_col_val))
        print("%Y-%m%d %H:%M:%S", time.localtime())

        # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
        save_result_to_csv(report, F1_score, experiment_name, model_name)

    print('all F1_score:', F1_scores / len(y_cols_name))

    print(">>>end of train_cnn_model function in featureFusion.py。。。")


# 训练模型,origin_data中包含多个属性的标签
def trainModel(experiment_name, model, x, embeddings_path, y, y_cols, ratio_style, model_name, epoch=5, batch_size=64, debug=False):
    print(">>>勿扰！训练模型ing...")
    print(">>>x's type = ", type(x))
    print(">>>y's type = ", type(y))

    F1_scores = 0
    F1_score = 0
    result = {}
    # if debug:
    #     y_cols = ['location']
    for index, col in enumerate(y_cols):
        experiment_name_aspect = experiment_name + "_" + col
        origin_data_current_col = y[col] + 2
        origin_data_current_col = np.array(origin_data_current_col)
        # 生成测试集,比例为0.1,x为numpy类型，origin_data为dataFrame类型
        ratio = 0.3
        length = int(len(origin_data_current_col) * ratio)
        print("测试集的长度为", length)
        x_validation = x[:length]
        x_train = x[length:]

        y_validation = origin_data_current_col[:length]
        y_train = origin_data_current_col[length:]
        print("y_train = ", y_train)

        print("origin_data_current.shape = ", origin_data_current_col.shape)
        print("origin_data_current_col = ", origin_data_current_col)

        print(">>>x_train.shape = ", x_train.shape)
        print(">>>x_validation.shape = ", x_validation.shape)

        # y_train_onehot = to_categorical(y_train)
        y_validation_onehot = to_categorical(y_validation)

        print(">>>y_train.shape = ", y_train.shape)
        print(">>>y_validation.shape = ", y_validation.shape)

        history = model.fit(dp.generateTrainSet(x_train, y_train, batch_size), validation_data=(x_validation, y_validation_onehot), epochs=epoch, verbose=2)

        # 预测验证集
        y_validation_pred = model.predict(x_validation)
        y_validation_pred = np.argmax(y_validation_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        print("y_val_pred = ", list(y_validation_pred))
        precision, recall, fscore, support = score(y_validation, y_validation_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        report = classification_report(y_validation, y_validation_pred, digits=4, output_dict=True)
        print(report)

        F1_score = f1_score(y_validation_pred, y_validation, average='macro')
        F1_scores += F1_score
        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_validation_pred, y_validation))
        print("%Y-%m%d %H:%M:%S", time.localtime())

        # 保存当前属性的结果,整体的结果根据所有属性的结果来计算
        save_result_to_csv(report, F1_score, experiment_name_aspect, model_name)

    print('all F1_score:', F1_scores / len(y_cols))

    print(">>>end of trainModel function...in absa_models...")


# 把结果保存到csv
# report是classification_report生成的字典结果
def save_result_to_csv(report, f1_score, experiment_id, model_name):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [experiment_id, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, f1_score, accuracy]

    path = "result/result_absa_" + model_name + ".csv"
    with codecs.open(path, "a", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


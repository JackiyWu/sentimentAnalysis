#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import codecs

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import keras
from keras.layers import Input, Flatten, Dense, Dropout, Activation, GRU, Bidirectional, Conv1D
from keras.layers import Embedding, merge, Lambda, Reshape, BatchNormalization, MaxPool1D, GlobalAveragePooling1D
from keras import Model, Sequential
from keras.utils import to_categorical
from keras import regularizers
from keras.layers.merge import concatenate

from tensorflow.keras.layers import SeparableConvolution1D

from keras_bert import Tokenizer, load_trained_model_from_checkpoint

import absa_config as config

# 一些超参数
TOKEN_DICT = {}


# 创建bert+fc模型
def createBertFcModel():
    pass


# 创建bert模型
def createBertEmbeddingModel():
    with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            TOKEN_DICT[token] = len(TOKEN_DICT)

    model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path)

    return model


# 根据textCNN模型输出词向量
def createTextCNNModel(maxlen, embedding_dim, debug=False):
    if debug:
        embedding_dim = 8
    # print(">>>开始构建TextCNN模型。。。")
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


# 构建模型CNN+BiGRU
def createTextCNNBiGRUModel(maxlen, embedding_dim, debug=False):
    if debug:
        embedding_dim = 8
    # print(">>>开始构建TextCNNBiGRUModel模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn1 = SeparableConvolution1D(200, 3, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_0")(tensor_input)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPool1D(pool_size=100)(cnn1)
    cnn2 = SeparableConvolution1D(200, 4, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_1")(tensor_input)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPool1D(pool_size=100)(cnn2)
    cnn3 = SeparableConvolution1D(200, 5, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001), name="separable_conv1d_2")(tensor_input)
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


# 训练模型,origin_data中包含多个属性的标签
def trainModel(model, x, origin_data, y_cols, ratio_style, epoch=3, batch_size=16, debug=False):
    print(">>>勿扰！训练模型ing...")
    print(">>>x's type = ", type(x))
    print(">>>origin_data's type = ", type(origin_data))
    origin_data_content = np.array(origin_data['content'])

    F1_scores = 0
    F1_score = 0
    result = {}
    # if debug:
    #     y_cols = ['location']
    for index, col in enumerate(y_cols):
        origin_data_current_col = origin_data[col] + 2
        origin_data_current_col = np.array(origin_data_current_col)
        # 生成测试集,比例为0.1,x为numpy类型，origin_data为dataFrame类型
        ratio = 0.1
        length = int(len(origin_data_current_col) * ratio)
        print("测试集的长度为", length)
        x_test = x[:length]
        x_current = x[length:]
        x_test_content = origin_data_content[:length]

        y_test = origin_data_current_col[:length]
        origin_data_current_col = origin_data_current_col[length:]

        # print("x = ", x)
        # print("origin_data_current.shape = ", origin_data_current_col.shape)
        # print("origin_data_current_col = ", origin_data_current_col)
        # print("origin_data = ", origin_data[col])
        # print("origin_data[content] = ", origin_data["content"])
        if ratio_style:
            x_train, x_validation, y_train, y_validation = train_test_split(x_current, origin_data_current_col, test_size=0.3)

        print(">>>x_train.shape = ", x_train.shape)
        print(">>>x_validation.shape = ", x_validation.shape)

        y_train_onehot = to_categorical(y_train)
        y_validation_onehot = to_categorical(y_validation)

        print(">>>y_train.shape = ", y_train.shape)
        print(">>>y_validation.shape = ", y_validation.shape)

        history = model.fit(x_train, y_train_onehot, validation_data=(x_validation, y_validation_onehot), epochs=epoch, batch_size=batch_size)

        # 预测验证集
        y_validation_pred = model.predict(x_validation)
        y_validation_pred = np.argmax(y_validation_pred, axis=1)

        # 预测并打印测试集
        y_test_pred = model.predict(x_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        length_test = len(y_test)
        # for i in range(length_test):
        #     print(origin_data_content[i]+" : realLabel-", y_test[i], ",predictedLabel-", y_test_pred[i])

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        print("y_val_pred = ", list(y_validation_pred))
        precision, recall, fscore, support = score(y_validation, y_validation_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        print(classification_report(y_validation, y_validation_pred, digits=4))

        F1_score = f1_score(y_validation_pred, y_validation, average='macro')
        F1_scores += F1_score

        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_validation_pred, y_validation))
        print("%Y-%m%d %H:%M:%S", time.localtime())

    print('all F1_score:', F1_scores/len(y_cols))
    print("result:", result)

    return result


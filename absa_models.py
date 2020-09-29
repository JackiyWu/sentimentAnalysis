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
from keras.layers import Input, Flatten, Dense, Dropout, Activation, GRU, Bidirectional, Conv1D, SeparableConvolution1D
from keras.layers import Embedding, merge, Lambda, Reshape, BatchNormalization, MaxPool1D, GlobalAveragePooling1D
from keras import Model, Sequential
from keras.utils import to_categorical
from keras import regularizers
from keras.layers.merge import concatenate

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
def createTextCNNModel(maxlen, embedding_dim):
    # print(">>>开始构建TextCNN模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn1 = SeparableConvolution1D(200, 3, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001))(tensor_input)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPool1D(pool_size=100)(cnn1)
    cnn2 = SeparableConvolution1D(200, 4, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001))(tensor_input)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPool1D(pool_size=100)(cnn2)
    cnn3 = SeparableConvolution1D(200, 5, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001))(tensor_input)
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
def createTextCNNBiGRU(maxlen, embedding_dim):
    # print(">>>开始构建TextCNNBiGRU模型。。。")
    tensor_input = Input(shape=(maxlen, embedding_dim))
    cnn1 = SeparableConvolution1D(200, 3, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001))(tensor_input)
    cnn1 = BatchNormalization()(cnn1)
    cnn1 = MaxPool1D(pool_size=100)(cnn1)
    cnn2 = SeparableConvolution1D(200, 4, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001))(tensor_input)
    cnn2 = BatchNormalization()(cnn2)
    cnn2 = MaxPool1D(pool_size=100)(cnn2)
    cnn3 = SeparableConvolution1D(200, 5, padding='same', strides=1, activation='relu', kernel_regularizer=regularizers.l1(0.00001))(tensor_input)
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

    # print(">>>TextCNNBiGRU模型构建结束。。。")
    return model


# 训练模型,origin_data中包含多个属性的标签
def trainModel(model, x, origin_data, epoch=3, batch_size=64, debug=False):
    F1_scores = 0
    F1_score = 0
    result = {}
    if debug:
        y_cols = ['location']


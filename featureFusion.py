#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tensorflow.keras.layers import Flatten, Input, Dense, Dropout, concatenate, Activation, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Conv1D, Conv2D, MaxPool1D, MaxPool2D, Embedding, GlobalAveragePooling1D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from keras.utils.np_utils import to_categorical
import keras

import numpy as np
import os
import time

import dataProcess as dp
import config


'''
# 一些超参数
max_words = 50000  # 词典的长度，后期可以测试？？
maxlen = 0  # 样本的长度，后期可以测试不同长度对结果的影响，参考hotelDataEmbedding.py的处理？？
embedding_dim = 128  # 词嵌入的维度，后期可以测试？？
embeddings = []  # 词嵌入，后期可以使用预训练词向量??
dealed_train = []  # 输入语料，训练集
dealed_val = []  # 输入语料，验证集
dealed_test = []  # 输入语料，测试集
y_cols = []
'''


# 创建融合模型
# filters格式为列表，如[64, 32]，方便后续调优
def create_fusion_model(fuzzy_maxlen, cnn_maxlen, dict_length, filters):
# def create_fusion_model(fuzzy_maxlen, cnn_maxlen, dict_length, filters, embedding_matrix):
    print("in create_fusion_model function...in featureFusion.py...")
    # define our MLP network
    inputs_fuzzy = Input(shape=(fuzzy_maxlen,), name='input_fuzzy')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
    x_fuzzy = Dense(16, activation='linear', name='dense1_fuzzy')(inputs_fuzzy)
    # x_mlp = Dense(4, activation='relu', name='dense2_mlp')(x_mlp)

    # define our CNN
    chanDim = -1
    inputs_cnn = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
    x_cnn = Embedding(input_dim=dict_length, output_dim=128, name='embedding_cnn')(inputs_cnn)
    for (i, filter) in enumerate(filters):
        x_cnn = Conv1D(filter, 3, activation='relu', name='conv' + str(i + 1))(x_cnn)
        # x_cnn = BatchNormalization(axis=chanDim)(x_cnn)
        x_cnn = MaxPool1D(name='pool' + str(i + 1))(x_cnn)
    x_cnn = Flatten(name='flatten')(x_cnn)

    # 融合两个输入
    x_concatenate = concatenate([x_fuzzy, x_cnn], name='fusion')

    x = Dense(32, activation='relu', name='dense3')(x_concatenate)
    x = Dropout(0.5, name="dropout1")(x)
    x = Dense(4, activation='softmax', name='softmax')(x)

    fusion_model = Model(inputs=[inputs_fuzzy, inputs_cnn], outputs=x, name='fusion_model')

    print(fusion_model.summary())

    # return our model
    return fusion_model


# 训练模型
def train_model(model, train, val, train_x_fuzzy, train_x_cnn, test_x_fuzzy, test_x, val_x_fuzzy, val_x, y_cols, class_weights, epoch, debug=False, folds=1):
    print(">>>in train_model function...")

    # print(mlp_model.summary())
    print("检查输入数据：")
    print("train_x_fuzzy.shape = ", train_x_fuzzy.shape)
    print("test_x.shape = ", test_x.shape)
    print("val_x.shape = ", val_x.shape)
    # print("y_cols_mlp = ", y_cols)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    F1_scores = 0
    F1_score = 0
    result = {}
    if debug:
        y_cols = ['location']
    # print("train = ", train)
    for index, col in enumerate(y_cols):
        # 只看第一个细粒度
        # if index >= 1:
        #     break

        train_y = train[col] + 2
        val_y = val[col] + 2
        y_val_pred = 0
        y_test_pred = 0
        #         epochs=[5,10]   , stratify=train_y
        # print("train_x_fuzzy.shape:", train_x_fuzzy.shape)
        # print("train_x_cnn.shape:", train_x_cnn.shape)

        print("train_y.shape:", train_y.shape)
        print("val_x.shape:", val_x.shape)
        print("val_y.shape:", val_y.shape)

        # 计算class_weight
        # cw = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
        cw = class_weights[index]

        for i in range(folds):
            y_train_onehot = to_categorical(train_y)
            y_val_onehot = to_categorical(val_y)
            # print("train_x_fuzzy = ", train_x_fuzzy)
            # print("train_x_fuzzy's shape = ", train_x_fuzzy.shape)
            # print("train_x_cnn = ", train_x_cnn)
            # print("train_x_cnn's shape = ", train_x_cnn.shape)
            history = model.fit([train_x_fuzzy, train_x_cnn], y_train_onehot, epochs=epoch, verbose=2,
                                batch_size=128, validation_data=([val_x_fuzzy, val_x], y_val_onehot),
                                class_weight=cw, callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.00001)])
                                # callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.00001)])

            # 预测验证集和测试集
            y_val_pred = model.predict([val_x_fuzzy, val_x])
            y_test_pred += model.predict([test_x_fuzzy, test_x])

        y_val_pred = np.argmax(y_val_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        # print("val_y = ", val_y)
        print("y_val_pred = ", list(y_val_pred))
        precision, recall, fscore, support = score(val_y, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        print(classification_report(val_y, y_val_pred, digits=4))

        F1_score = f1_score(y_val_pred, val_y, average='weighted')
        F1_scores += F1_score

        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred-2
    print('all F1_score:', F1_scores/len(y_cols))
    print("result:", result)

    print(">>>end of train_mlp function...")

    return result


# cnn
# def create_cnn_model(cnn_maxlen, dict_length, filters, embedding_matrix):
def create_cnn_model(cnn_maxlen, dict_length, filters, embedding_matrix):
    inputs_cnn = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
    x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_cnn)
    # x_cnn = Embedding(input_dim=dict_length, output_dim=200, name='embedding_cnn')(inputs_cnn)
    for (i, filter) in enumerate(filters):
        x_cnn = Conv1D(filter, 3, activation='relu', name='conv' + str(i + 1))(x_cnn)
        # x_cnn = BatchNormalization(axis=chanDim)(x_cnn)
        x_cnn = MaxPool1D(name='pool' + str(i + 1))(x_cnn)
    x_cnn = Flatten(name='flatten')(x_cnn)

    x = Dense(32, activation='relu', name='dense3')(x_cnn)
    x = Dropout(0.5, name="dropout1")(x)
    x = Dense(4, activation='softmax', name='softmax')(x)

    cnn_model = Model(inputs=inputs_cnn, outputs=x, name='cnn_model')

    print(cnn_model.summary())

    # return our model
    return cnn_model


# train cnn
def train_cnn_model(model, train, val, train_x, test_x, val_x, y_cols, class_weights, epoch, debug=False, folds=1):
    print(">>>in train_cnn_model function。。。")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    F1_scores = 0
    F1_score = 0
    result = {}
    if debug:
        y_cols = ['location']
    # print("train = ", train)
    for index, col in enumerate(y_cols):
        # 只看第一个细粒度
        # if index >= 1:
        #     break

        train_y = train[col] + 2
        val_y = val[col] + 2
        y_val_pred = 0
        y_test_pred = 0

        print("train_y.shape:", train_y.shape)
        print("val_x.shape:", val_x.shape)
        print("val_y.shape:", val_y.shape)

        # 计算class_weight
        # cw = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
        cw = class_weights[index]

        for i in range(folds):
            y_train_onehot = to_categorical(train_y)
            y_val_onehot = to_categorical(val_y)
            history = model.fit(train_x, y_train_onehot, epochs=epoch, verbose=2,
                                batch_size=128, validation_data=(val_x, y_val_onehot),
                                # class_weight=cw)
                                # class_weight=cw, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)])
                                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)])

            # 预测验证集和测试集
            y_val_pred = model.predict(val_x)
            y_test_pred += model.predict(test_x)

        y_val_pred = np.argmax(y_val_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        # print("y_val_pred = ", list(y_val_pred))
        precision, recall, fscore, support = score(val_y, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        print(classification_report(val_y, y_val_pred, digits=4))

        F1_score = f1_score(y_val_pred, val_y, average='weighted')
        F1_scores += F1_score

        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred-2
    print('all F1_score:', F1_scores/len(y_cols))
    print("result:", result)

    print(">>>end of train_cnn_model function in featureFusion.py。。。")

    return result


# 生成预训练词向量,size-
def load_word2vec(word_index):
    print(">>>in load_word2vec function of featureFusion.py...")
    print("word_index's lengh = ", len(word_index))
    f = open(config.pre_word_embedding, "r", encoding="utf-8")
    length, dimension = f.readline().split()  # 预训练词向量的单词数和词向量维度
    dimension = int(dimension)
    print("length = ", length, ", dimension = ", dimension)

    # 创建词向量索引字典
    embeddings_index = {}

    print(">>>读取预训练词向量ing。。。")

    for line in f:
        # print("line = ", line)
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
        # print(word, ":", coefs)
    f.close()

    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    # 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))
    # 遍历词汇表中的每一项
    for word, i in word_index.items():
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        # print("embedding_vector = ", embedding_vector)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # print("embedding_matrix = ", embedding_matrix)
    print(">>>end of load_word2vec function in featureFusion.py...")

    return embedding_matrix


# fasttext model
def fasttext_model(fea_dict, maxlen):
    inputs_fasttext = Input(shape=(maxlen,), name='fasttext_input')
    x_fasttext = Embedding(input_dim=len(fea_dict), output_dim=200, name='embedding_fasttext')(inputs_fasttext)
    x_fasttext = GlobalAveragePooling1D(name='GlobalAveragePooling1D_fasttext')(x_fasttext)
    x_fasttext = Dropout(0.3)(x_fasttext)
    x_fasttext = Dense(4, activation='softmax', name='softmax')(x_fasttext)

    model = Model(inputs=inputs_fasttext, outputs=x_fasttext, name='fasttext_model')

    print(model.summary())

    return model


# train fasttext model
def train_fasttext_model(model, train, val, train_x, test_x, val_x, y_cols, class_weights, epoch, debug=False, folds=1):
    print(">>>in train_fasttext_model function。。。")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    F1_scores = 0
    F1_score = 0
    result = {}
    if debug:
        y_cols = ['location']
    # print("train = ", train)
    for index, col in enumerate(y_cols):
        # 只看第一个细粒度
        # if index >= 1:
        #     break

        train_y = train[col] + 2
        val_y = val[col] + 2
        y_val_pred = 0
        y_test_pred = 0

        print("train_y.shape:", train_y.shape)
        print("val_x.shape:", val_x.shape)
        print("val_y.shape:", val_y.shape)

        # 计算class_weight
        # cw = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
        # cw = class_weights[index]

        for i in range(folds):
            y_train_onehot = to_categorical(train_y)
            y_val_onehot = to_categorical(val_y)
            history = model.fit(train_x, y_train_onehot, epochs=epoch,
                                batch_size=64, validation_data=(val_x, y_val_onehot),
                                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)])
                                # class_weight=cw)

            # 预测验证集和测试集
            y_val_pred = model.predict(val_x)
            y_test_pred += model.predict(test_x)

        y_val_pred = np.argmax(y_val_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        print("y_val_pred = ", list(y_val_pred))
        precision, recall, fscore, support = score(val_y, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        print(classification_report(val_y, y_val_pred, digits=4))

        F1_score = f1_score(y_val_pred, val_y, average='macro')
        F1_scores += F1_score

        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
        print("%Y-%m%d %H:%M:%S", time.localtime())
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred-2
    print('all F1_score:', F1_scores/len(y_cols))
    print("result:", result)

    print(">>>end of train_fasttext_model function in featureFusion.py。。。")

    return result


# lstm模型
def create_lstm_model(maxlen, dict_length):
# def create_lstm_model(maxlen, dict_length, embedding_matrix):
    inputs_lstm = Input(shape=(maxlen,), name='inputs_lstm')

    embedding = Embedding(input_dim=dict_length, output_dim=200, name='embedding_cnn')(inputs_lstm)

    x_lstm = LSTM(32)(embedding)
    # x_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    # x_lstm = Bidirectional(LSTM(64, return_sequences=False))(x_lstm)
    x_lstm = Dense(32, activation='relu', name='FC1')(x_lstm)
    x_lstm = Dropout(0.5)(x_lstm)
    x_lstm = Dense(4, activation='softmax', name='FC2')(x_lstm)

    model = Model(inputs=inputs_lstm, outputs=x_lstm, name='lstm_model')

    print(model.summary())

    return model


# 全连接神经网络，测试情感向量
def fnn_model(input_shape, input_shape2):
    # define our MLP network
    inputs_fnn = Input(shape=(input_shape,), name='input_fnn')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
    x_fnn = Dense(4, activation='relu', name='dense1_fnn')(inputs_fnn)
    x_fnn = Dense(2, activation='linear', name='dense2_fnn')(x_fnn)

    # x = Dense(2, activation='softmax', name='softmax')(x_fnn)

    inputs_fnn2 = Input(shape=(input_shape2,), name='input_fnn2')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
    x_fnn2 = Dense(4, activation='relu', name='dense1_fnn2')(inputs_fnn2)
    x_fnn2 = Dense(3, activation='linear', name='dense2_fnn2')(x_fnn2)

    # 融合两个输入
    x_concatenate = concatenate([x_fnn, x_fnn2], name='fusion')

    x = Dense(32, activation='relu', name='dense3')(x_concatenate)
    x = Dropout(0.5, name="dropout1")(x)
    x = Dense(2, activation='softmax', name='softmax')(x)

    fusion_model = Model(inputs=[inputs_fnn, inputs_fnn2], outputs=x, name='fusion_model')

    print(fusion_model.summary())

    # return our model
    return fusion_model


'''
if __name__ == "__main__":
    print(">>>begin execute featureFusion.py...")

    origin_data, y_cols = dp.initData()
    stoplist = dp.getStopList()
    dealed_train, dealed_val, dealed_test, train, val, test, texts = dp.processData(origin_data, stoplist, 5000)

    # print(origin_data.head())
    # print(stoplist)
    # print("dealed_train:", train)
    # print("dealed_val:", val)
    # print("dealed_test:", test)
    print("*" * 200)

    model = create_fusion_model(512, 5000)

    plot_model(model, 'modelsImage/Multi_input_model.png')

    # print(model.summary())

    dealed_train_mlp, dealed_train_cnn = dealed_train, dealed_train

    train_model(model, dealed_train_mlp, dealed_train_cnn, dealed_test, dealed_val, y_cols)

    print(">>>end of featureFusion.py...")
'''


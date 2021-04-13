#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Input, Dense, Dropout, concatenate, Activation, LSTM, Bidirectional, GRU
from tensorflow.keras.layers import BatchNormalization, Conv1D, Conv2D, MaxPool1D, MaxPool2D, Embedding, GlobalAveragePooling1D
from tensorflow.keras import Model, Sequential
from tensorflow.keras import optimizers
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import KFold

import numpy as np
import sys

import auto_usefulness_dataProcess as dp

pre_word_embedding = "config/preEmbeddings/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"


# 贝叶斯分类器
def bayesModel():
    classifier = GaussianNB()

    return classifier


# AdaBoost分类器
def adaBoostModel():
    classifier = AdaBoostClassifier(n_estimators=10)

    return classifier


# SVM
def svmModel():
    classifier = Pipeline((("scaler", StandardScaler()), ("liner_svc", LinearSVC(C=1, loss="hinge")), ))

    return classifier


# 逻辑回归
def logicRegressionModel():
    classifier = LogisticRegression(C=1e5)

    return classifier


# 决策树
def decisionTreeModel():
    clf = DecisionTreeClassifier(criterion="entropy")

    return clf


# 随机森林
def randomForestModel():
    clf = RandomForestClassifier()

    return clf


# 生成预训练词向量,size-
def load_word2vec(word_index):
    print(">>>in load_word2vec function of featureFusion.py...")
    print("word_index's lengh = ", len(word_index))
    f = open(pre_word_embedding, "r", encoding="utf-8")
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


# 构建模型 单纯使用CNN
def createCNNModel(contents_length, dict_length, embedding_matrix):
    window_size = 5

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_cnn = Input(shape=(contents_length,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_cnn)
        x_cnn = Conv1D(64, window_size, activation='relu', name='conv1')(x_cnn)
        x_cnn = MaxPool1D(name='pool1')(x_cnn)
        x_cnn = Flatten(name='flatten')(x_cnn)

        x = Dense(128, activation='relu', name='dense3')(x_cnn)
        x = Dropout(0.5, name="dropout2")(x)
        x = Dense(2, activation='softmax', name='softmax')(x)

        model = Model(inputs_cnn, outputs=x, name='final_model')

        adam = optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(model.summary())

    return model


# 构建模型
def createFusionVectorsModel(language_feature_length, contents_length, dict_length, embedding_matrix, language_feature_node, cnn_node):
    print("language_feature_length = ", language_feature_length)
    print("contents_length = ", contents_length)
    print("dict_length = ", dict_length)
    print("embedding_matrix = ", embedding_matrix)
    window_size = 5
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_language_feature = Input(shape=(language_feature_length,), name='input_language_feature')
        x_language_feature = Dense(language_feature_node, activation='linear', name='dense1_language_feature')(input_language_feature)
        print("x_languagee_feature's type = ", type(x_language_feature))

        # define our CNN
        inputs_cnn = Input(shape=(contents_length,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        # x_cnn = Embedding(input_dim=dict_length, output_dim=128, name='embedding_cnn')(inputs_cnn)
        x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_cnn)
        x_cnn = Conv1D(cnn_node, window_size, activation='relu', name='conv1')(x_cnn)
        # x_cnn = BatchNormalization(axis=chanDim)(x_cnn)
        x_cnn = MaxPool1D(name='pool1')(x_cnn)
        x_cnn = Flatten(name='flatten')(x_cnn)
        print("x_cnn's type = ", type(x_cnn))

        # 融合两个输入
        x_concatenate = concatenate([x_language_feature, x_cnn], axis=-1, name='fusion')

        x = Dense(128, activation='relu', name='dense3')(x_concatenate)
        # x = Dropout(0.4, name="dropout1")(x)
        # x = Dense(32, activation='relu', name='dense4')(x)
        x = Dropout(0.5, name="dropout2")(x)
        x = Dense(2, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=[input_language_feature, inputs_cnn], outputs=x, name='fusion_model')

        adam = optimizers.Adam(learning_rate=0.001)

        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(fusion_model.summary())

    return fusion_model


# 构建模型LSTM,输入只有语义特征向量
def createLSTMModel(contents_length, dict_length, embedding_matrix, dim):
    print("contents_length = ", contents_length)
    print("dict_length = ", dict_length)
    print("embedding_matrix = ", embedding_matrix)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_contents = Input(shape=(contents_length,), name="input_contents")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        x_contents = Embedding(input_dim=dict_length, output_dim=300, name='embedding_contents', weights=[embedding_matrix], trainable=True)(input_contents)

        x_contents = LSTM(dim, return_sequences=False, name='lstm')(x_contents)

        x = Dropout(0.5, name="dropout2")(x_contents)
        x = Dense(2, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=input_contents, outputs=x, name='fusion_model')

        adam = optimizers.Adam(learning_rate=0.001)

        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(fusion_model.summary())

    return fusion_model


# 构建模型GRU,输入只有语义特征向量
def createGRUModel(contents_length, dict_length, embedding_matrix, dim):
    print("contents_length = ", contents_length)
    print("dict_length = ", dict_length)
    print("embedding_matrix = ", embedding_matrix)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_contents = Input(shape=(contents_length,), name="input_contents")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        x_contents = Embedding(input_dim=dict_length, output_dim=300, name='embedding_contents', weights=[embedding_matrix], trainable=True)(input_contents)

        x_contents = GRU(dim, name='gru')(x_contents)

        x = Dropout(0.5, name="dropout2")(x_contents)
        x = Dense(2, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=input_contents, outputs=x, name='fusion_model')

        adam = optimizers.Adam(learning_rate=0.001)

        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(fusion_model.summary())

    return fusion_model


# 构建模型MLP，输入只有语义特征向量
def createMLPModel(contents_length, dict_length, embedding_matrix, contents_node):
    print("contents_length = ", contents_length)
    print("dict_length = ", dict_length)
    print("embedding_matrix = ", embedding_matrix)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        input_contents = Input(shape=(contents_length,), name="input_contents")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        x_contents = Embedding(input_dim=dict_length, output_dim=300, name='embedding_contents', weights=[embedding_matrix], trainable=True)(input_contents)
        x_contents = Flatten(name='flatten')(x_contents)
        print("x_cnn's type = ", type(x_contents))

        x_contents = Dense(contents_node, activation='relu', name='dense3')(x_contents)

        x = Dropout(0.5, name="dropout2")(x_contents)
        x = Dense(2, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=input_contents, outputs=x, name='mlp_model')

        adam = optimizers.Adam(learning_rate=0.001)

        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(fusion_model.summary())

    return fusion_model


# 训练模型
def trainModel(model_name, current_model, X_language_feature, X_contents, Y, epoch, batch_size, debug=False):
    print(">>>in trainModel function...")

    if len(X_language_feature) != len(X_contents) or len(X_language_feature) != len(Y):
        print(">>>>出错辣！！！喂给模型的数据长度不一致!!!")
        sys.exit(-1)

    print("X_language_feature's type = ", type(X_language_feature))
    print("X_contents's type = ", type(X_contents))
    print("Y's type = ", type(Y))
    print("X_language_feature's shape = ", X_language_feature.shape)
    print("X_contents's shape = ", X_contents.shape)
    print("Y's shape = ", Y.shape)

    kf = KFold(n_splits=10)
    current_k = 0
    for train_index, validation_index in kf.split(X_contents):
        print("正在进行第", current_k, "轮交叉验证。。。")
        current_k += 1
        X_language_feature_train, Y_train = X_language_feature[train_index], Y[train_index]
        X_contents_train, Y_train = X_contents[train_index], Y[train_index]
        Y_train = to_categorical(Y_train)
        X_language_feature_validation, Y_validation = X_language_feature[validation_index], Y[validation_index]
        X_contents_validation, Y_validation = X_contents[validation_index], Y[validation_index]
        Y_validation_onehot = to_categorical(Y_validation)

        if model_name.startswith("fusion"):
            current_model.fit([X_language_feature_train, X_contents_train], Y_train, epochs=epoch, verbose=2, batch_size=batch_size,
                              validation_data=([X_language_feature_validation, X_contents_validation], Y_validation_onehot))
            predicts = current_model.predict([X_language_feature_validation, X_contents_validation])
        else:
            current_model.fit(X_contents_train, Y_train, epochs=epoch, verbose=2, batch_size=batch_size,
                              validation_data=(X_contents_validation, Y_validation_onehot))
            predicts = current_model.predict(X_contents_validation)

        predicts = np.argmax(predicts, axis=1)
        print("predicts' type = ", type(predicts))

        # 计算各种评价指标&保存结果
        dp.calculateScore(Y_validation.tolist(), predicts, model_name, debug)


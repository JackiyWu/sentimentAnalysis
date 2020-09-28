#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import re
import csv
from collections import Counter
import codecs
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from keras.utils.np_utils import to_categorical
import keras

import absa_config as config
import numpy as np
import os
import time
import codecs

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.cluster import KMeans

import keras
from keras.utils.np_utils import to_categorical

from keras_bert import Tokenizer as bert_Tokenizer, load_trained_model_from_checkpoint

import absa_config as config


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


# 读取数据
def initData(debug=False):
    # print("In initData function of dataProcess.py...")
    data = pd.read_csv(config.meituan_validation_new)
    # data = pd.read_csv(config.meituan_train)
    if debug:
        data = data[:10]

    global y_cols
    # y_cols = data.columns.values.tolist()[2:22]
    y_cols = data.columns.values.tolist()[2:7]
    print("data = ", data.head())
    print("data'type = ", type(data))
    print("data's shape = ", data.shape)
    print("y_cols = ", y_cols)

    # print("end of initData function in dataProcess.py...")

    return data, y_cols


# 输入细粒度的源数据，生成包括6个粗粒度属性的数据，存入csv
# origin_data的2-21列是分细粒度属性的标签列。2-4：位置location，5-8：服务service，9-11：价格price，12-15：环境environment，16-18：菜品dish，19-21：其他others
# 计算规则：①
def processDataToTarget(origin_data):
    print("in processDataToTarget function of dataProcess.py...")
    header = ["id", "content", "location", "service", "price", "environment", "dish"]
    # header = ["id", "content", "location", "service", "price", "environment", "dish", "others"]

    data = []

    for index, row in origin_data.iterrows():
        d = [row["id"], row["content"]]

        location = [row["location_traffic_convenience"], row["location_distance_from_business_district"], row["location_easy_to_find"]]
        service = [row["service_wait_time"], row["service_waiters_attitude"], row["service_parking_convenience"], row["service_serving_speed"]]
        price = [row["price_level"], row["price_cost_effective"], row["price_discount"]]
        environment = [row["environment_decoration"], row["environment_noise"], row["environment_space"], row["environment_cleaness"]]
        dish = [row["dish_portion"], row["dish_taste"], row["dish_look"], row["dish_recommendation"]]
        # others = [row["others_overall_experience"], row["others_willing_to_consume_again"]]

        location = processLabel(location)
        service = processLabel(service)
        price = processLabel(price)
        environment = processLabel(environment)
        dish = processLabel(dish)
        # others = processLabel(others)

        d.append(location)
        d.append(service)
        d.append(price)
        d.append(environment)
        d.append(dish)
        # d.append(others)

        data.append(d)
        # print("d = ", d)
    # print("data = ", data)
    with codecs.open(config.meituan_validation_new, "wb", "utf-8") as f:
    # with codecs.open(config.meituan_train_new, "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
        f.close()

    print("end of processDataToTarget function in dataProcess.py...")


# 传入细粒度属性的label，输出粗粒度属性label
def processLabel(aspect):
    # print("aspect = ", aspect)
    if -2 not in aspect:  # 如果标签没有-2，则求整体平均
        average = np.average(np.array(aspect))
    else:  # label中有-2分为两种情况
        length = len(set(aspect))

        if length > 1:  # 不全是-2
            while -2 in aspect:
                aspect.remove(-2)
            average = np.average(aspect)
        else:  # 全是-2
            average = -2

    # 如果average非整数&≠-2
    if average != -2:
        if average > 0:
            average = 1
        elif average < 0:
            average = -1
        else:
            average = 0

    # print("average = ", average)
    return average


# 根据类别label数量，计算得到class_weight
def calculate_class_weight(labels):
    print(">>>in calculate_class_weights function of dataProcess.py...")
    print("labels = ", labels)
    class_weights = []
    for label in labels:
        max_value = max(label.values())
        print(label.get(-2), label.get(-1), label.get(0), label.get(1))
        class_weights.append({0: max_value / label.get(-2), 1: max_value / label.get(-1), 2: max_value / label.get(0), 3: max_value / label.get(1)})

    print("class_weights = ", class_weights)

    print(">>>end of calculate_class_weights function in dataProcess.py...")

    return class_weights


def getStopList():
    stoplist = pd.read_csv('stopwords.txt').values
    return stoplist


# 处理数据得到输入语料的文本
def processDataToTexts(data, stoplist, no_punctuation=True):
    print(">>>in processDataToTexts of dataProcess.py...")
    print("data['content'] = ", data['content'])

    data['words'] = data['content']

    # 去标点符号
    if no_punctuation:
        print(">>>去标点符号ing。。。")
        data['words'] = data['words'].apply(lambda x: re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？、~@#￥%……&*（）．；：】【|]+", "", x))
        # jieba分词
        print(">>>jieba分词ing。。。")
        data['words'] = data['words'].apply(lambda x: list(jieba.cut(x)))

    # 去掉开头
    data['words'] = data['words'].apply(lambda x: x[1:-1])
    data['len'] = data['words'].apply(lambda x: len(x))
    maxlen = data['len'].max()
    words_dict = []
    texts = []

    # 去掉停用词
    print(">>>去停用词ing in DataProcess.py...")
    for index, row in data.iterrows():
        line = [word.strip() for word in list(row['words']) if word not in stoplist]

        words_dict.extend([word for word in line])
        texts.append(line)

    print("texts = ", texts)

    print(">>>end of processDataToTexts in dataProcess.py...")

    return texts


# 处理数据生成训练集 验证集 测试集
def processData(data, stoplist, dict_length, maxlen, ratios):
    print(">>>in processData function...")

    # print(words_dict)
    # print(texts)
    texts = processDataToTexts(data, stoplist)

    stop_data = pd.DataFrame(texts)

    # 利用keras的Tokenizer进行onehot，并调整未等长数组
    tokenizer = Tokenizer(num_words=dict_length)
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index
    # print("word_index = ", word_index)

    data_w = tokenizer.texts_to_sequences(texts)
    data_T = sequence.pad_sequences(data_w, maxlen=maxlen)

    # 数据划分，重新划分为训练集，测试集和验证集
    data_length = data_T.shape[0]
    print("data_length = ", data_length)
    size_train = int(data_length * ratios[0])
    size_val = int(data_length * ratios[1])

    global dealed_train
    global dealed_val
    global dealed_test
    dealed_train = data_T[: size_train]
    dealed_val = data_T[size_train: (size_train + size_val)]
    dealed_test = data_T[(size_train + size_val):]

    global train  # 训练数据集，包括语料、标签、id、len（后期增加的）
    global val  # 验证数据集，包括语料、标签、id、len（后期增加的）
    global test  # 测试数据集，包括语料、标签、id、len（后期增加的）
    train = data[: size_train]
    val = data[size_train: (size_train + size_val)]
    test = data[(size_train + size_val):]
    # print(train.shape)
    # print(val.shape)
    # print(test.shape)

    # print(data.columns.values.tolist())
    print(">>>end of processData function...")

    return dealed_train, dealed_val, dealed_test, train, val, test, texts, word_index


# 确定maxlen
def calculate_maxlen(texts):
    maxlen = 0

    # 直接取最长的评论长度
    for line in texts:
        if maxlen < len(line):
            maxlen = len(line)
    '''

    # 取评论长度的平均值+两个评论的标准差（假设评论长度的分布满足正态分布，则maxlen可以涵盖95左右的样本）
    lines_length = [len(line) for line in texts]
    lines_length = np.array(lines_length)

    maxlen = np.mean(lines_length) + 2 * np.std(lines_length)
    maxlen = int(maxlen)
    '''

    return maxlen


# 传入语料数据，输出不同属性-label的样本数,依次为位置、服务、价格、环境、菜品、其他
def calculateSampleNumber(origin_data):
    aspects = ["location", "service", "price", "environment", "dish"]

    result = []
    for aspect in aspects:
        current = list(origin_data[aspect])
        # print(aspect, " = ", current)
        result.append(Counter(current))

    return result


def build_model(max_words, embedding_dim, maxlen):
    print(">>>Building the model...")
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(5))
    # model.add(Dropout(0.5))
    model.add(Conv1D(64, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    # model.add(layers.Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model


def build_model2(max_words, embedding_dim, maxlen):
    print(">>>Building the model...")
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    conv1 = Conv1D(64, 3, activation='relu')
    conv11 = Conv1D(64, 3, activation='relu')
    added = keras.layers.add([conv1, conv11])
    model.add(added)
    pooling1 = MaxPooling1D(5)
    model.add(pooling1)
    # model.add(Dropout(0.5))
    conv2 = Conv1D(64, 3, activation='relu')
    model.add(conv2)
    # model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    # model.add(layers.Dense(32, activation='relu'))
    dense = Dense(4, activation='softmax')
    model.add(dense)
    return model


def train_CNN(train_x, test_x, val_x, y_cols, debug=False, folds=1):
    print(">>>in train_CNN function...")
    # print(train_x, test_x, val_x, y_cols)
    # print("train:", train)
    # print("val:", val)
    # print("test:", test)
    model = build_model()

    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    F1_scores = 0
    F1_score = 0
    result = {}
    if debug:
        y_cols = ['location_traffic_convenience']
    for index, col in enumerate(y_cols):
        train_y = train[col] + 2
        val_y = val[col] + 2
        y_val_pred = 0
        y_test_pred = 0
        #         epochs=[5,10]   , stratify=train_y
        for i in range(folds):
            y_train_onehot = to_categorical(train_y)
            history = model.fit(train_x, y_train_onehot, epochs=3, batch_size=64, validation_split=0.2)

            # 预测验证集和测试集
            y_val_pred = model.predict(val_x)
            y_test_pred += model.predict(test_x)

        y_val_pred = np.argmax(y_val_pred, axis=1)

        F1_score = f1_score(y_val_pred, val_y, average='macro')
        F1_scores += F1_score

        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred-2
    print('all F1_score:', F1_scores/len(y_cols))
    print("result:", result)
    return result


# 创建bert模型
def createBertEmbeddingModel():
    print(">>>开始Bert模型。。。")
    token_dict = {}
    with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path)
    tokenizer = bert_Tokenizer(token_dict)

    print(">>>Bert模型加载结束。。。")

    return model, tokenizer, token_dict


# 根据bert模型和input_texts得到字符级向量和句子级向量
# 评论长度限制为512个字符，后续可以扩大看效果
def getBertEmbeddings(bert_model, tokenizer, origin_data, debug=False):
    # print(">>>获取bert字符级向量和句子级向量。。。")
    character_embeddings = []
    sentence_embeddings = []

    input_texts = origin_data["content"]

    for text in input_texts:
        current_embedding = []
        # print("text = ", text)
        # print("text's length = ", len(text))
        tokens = tokenizer.tokenize(text)
        indices, segments = tokenizer.encode(first=text, max_len=512)
        # print("indices = ", indices[:10])
        # print("segments = ", segments[:10])

        predicts = bert_model.predict([np.array([indices]), np.array([segments])])[0]
        # print("tokens' length = ", len(tokens))
        for i, token in enumerate(tokens):
            # 此处为了限制评论最长是512个字符，后续可以做实验扩大
            if i >= 512:
                break

            if debug:
                predicted = predicts[i].tolist()[:5]
            else:
                predicted = predicts[i].tolist()
            # print(token, predicted)
            current_embedding.append(predicted)
            # current_embedding.append(predicts[i].tolist())
            if token == "[CLS]":
                sentence_embeddings.append(predicted)
        character_embeddings.append(current_embedding)
    # print("character_embeddings[0] = ", character_embeddings[0])
    # print("sentence_embeddings[0] = ", sentence_embeddings[0])

    # print(">>>bert字符级向量和句子级向量GET。。。")

    return character_embeddings, sentence_embeddings


# 对输入的评论文本向量（一个向量表示一个句子）进行聚类，得到三个聚类中心，并写入文件
def getClusterCenters(sentence_embeddings):
    print(">>>In getClusterCenters of absa_dataProcess.py...")
    kMeans_model = trainModel(sentence_embeddings)

    # 查看预测样本的中心点
    clusters_centers = kMeans_model.cluster_centers_
    # print("clusters' centers:", clusters_centers)

    # 将聚类中心点写入文件
    with codecs.open(config.cluster_center, "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(clusters_centers)
        f.close()

    print(">>>End of getClusterCenters in absa_dataProcess.py...")

    return clusters_centers


# 从文件中读取聚类中心向量
def getClusterCenterFromFile():
    cluster_center = pd.read_csv(config.cluster_center, header=None).values.tolist()
    print("cluster_center = ", cluster_center)

    return cluster_center


# 接收sentence_embeddings，训练模型并返回
def trainModel(sentence_embeddings):
    # 创建KMeans 对象
    cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)
    # print("sentence_embeddings = ", sentence_embeddings)
    result = cluster.fit(sentence_embeddings)

    return result


# 计算评论中的字向量与聚类中心的隶属度（余弦距离）
def calculateMembershipDegree(cluster_centers, character_embeddings):
    # print(">>>开始计算评论文本的隶属度。。。")
    membership_degrees = []
    for character_embedding in character_embeddings:
        sentence_membership_degrees = []
        for character in character_embedding:
            word_membership_degrees = []
            for cluster_center in cluster_centers:
                result = calculateCosinValue(character, cluster_center)
                word_membership_degrees.append(result)
            sentence_membership_degrees.append(word_membership_degrees)
        membership_degrees.append(sentence_membership_degrees)

    # print(">>>评论文本的隶属度计算结束。。。")
    return membership_degrees


# 使用余弦距离计算两个向量间的相似度
def calculateCosinValue(vector1, vector2):
    # print("vector1 = ", vector1)
    # print("vector2 = ", vector2)
    vector1 = np.mat(vector1)
    vector2 = np.mat(vector2)
    num = float(vector1 * vector2.T)
    denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    # print("sim = ", sim)

    return sim


# 将assisted_vector拼接在main_vector后面
def concatenateVector(main_vector, assisted_vector):
    print("main_vector's length = ", len(main_vector))
    # print("assisted_vector = ", assisted_vector)
    print("assisted_vector's length = ", len(assisted_vector))

    final_word_embeddings = []
    length_1 = len(main_vector)
    length_2 = len(assisted_vector)
    if length_1 != length_2:
        print("字向量和隶属值向量长度不一致!!!")
        return None

    for i in range(length_1):
        main_vector_current = main_vector[i]
        assisted_vector_current = assisted_vector[i]
        # print("main_vector_current = ", main_vector_current)
        # print("main_vector_current's length = ", len(main_vector_current))
        # print("main_vector_current's type = ", type(main_vector_current))
        # print("assisted_vector_current = ", assisted_vector_current)
        # print("assisted_vector_current's length = ", len(assisted_vector_current))
        # print("assisted_vector_current's type = ", type(assisted_vector_current))

        final_word_embeddings.append(np.concatenate((main_vector_current, assisted_vector_current), axis=1).tolist())
        # print("final_word_embeddings = ", final_word_embeddings)

    return np.array(final_word_embeddings)


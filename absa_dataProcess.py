#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from collections import Counter
import pandas as pd
import sys
import math
import xlrd

import numpy as np
import codecs

from sklearn.cluster import KMeans

from keras_bert import Tokenizer as bert_Tokenizer, load_trained_model_from_checkpoint, extract_embeddings
from keras.utils import to_categorical

import absa_config as config


# 读取数据
def initData(debug=False, clean_enter=False, clean_space=False):
    # print("In initData function of dataProcess.py...")
    data = pd.read_csv(config.meituan_validation_new)
    # data = pd.read_csv(config.meituan_train)
    if debug:
        data = data[:300]

    # data = data[:1000]
    y = data[['location', 'service', 'price', 'environment', 'dish']]

    # 对原评论文本进行清洗，去回车符 去空格
    # print("data['content']_0 = ", data['content'])
    if clean_enter:
        data = dataCleanEnter(data)
    if clean_space:
        data = dataCleanSpace(data)
    # print("data['content']_1 = ", data['content'])

    # y_cols_name = data.columns.values.tolist()[2:22]
    y_cols_name = data.columns.values.tolist()[2:7]
    print(">>>data = ", data.head())
    print(">>>data'type = ", type(data))
    print(">>>data's shape = ", data.shape)
    print(">>>y_cols_name = ", y_cols_name)

    # print("end of initData function in dataProcess.py...")

    return data, y_cols_name, y


def initDataLabels(debug=False):
    # print("In initDataLabels function of dataProcess.py...")
    data_train = pd.read_csv(config.meituan_train_new)
    data_validation = pd.read_csv(config.meituan_validation_new)
    # data = pd.read_csv(config.meituan_train)
    if debug:
        data_train = data_train[:300]
        data_validation = data_validation[:200]

    y_train = data_train[['location', 'service', 'price', 'environment', 'dish']]
    y_validation = data_validation[['location', 'service', 'price', 'environment', 'dish']]

    # y_cols_name = data.columns.values.tolist()[2:22]
    y_cols_name = data_train.columns.values.tolist()[2:7]
    print(">>>data_train = ", data_train.head())
    print(">>>data_train'type = ", type(data_train))
    print(">>>data_train's shape = ", data_train.shape)
    print(">>>data_validation = ", data_validation.head())
    print(">>>data_validation'type = ", type(data_validation))
    print(">>>data_validation's shape = ", data_validation.shape)
    print(">>>y_cols_name = ", y_cols_name)

    return y_cols_name, y_train, y_validation


# 对原评论文本进行清洗,去回车符
def dataCleanEnter(data):
    ids = data['id']
    # print("ids = ", ids)

    for i in ids:
        # print("i = ", i)
        # print("data.loc[i, 'content']_0 = ", data.loc[i, 'content'])
        current = data.loc[i, 'content']
        # print("current_0 = ", current)
        current = current.replace('\n', '')
        # print("current_1 = ", current)
        data.loc[i, 'content'] = current
        # print("data.loc[i, 'content']_1 = ", data.loc[i, 'content'])

    return data


# 对原评论文本进行清洗，去空格
def dataCleanSpace(data):
    pass


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


# 传入语料数据，输出不同属性-label的样本数,依次为位置、服务、价格、环境、菜品、其他
def calculateSampleNumber(origin_data):
    aspects = ["location", "service", "price", "environment", "dish"]

    result = []
    for aspect in aspects:
        current = list(origin_data[aspect])
        # print(aspect, " = ", current)
        result.append(Counter(current))

    return result


# 如果文本长度小于maxlen，则进行[pad]补齐
def textsPadding(tokens, maxlen):
    length = len(tokens)
    if length < maxlen:
        pad = '[PAD]'
        tokens += [pad] * (maxlen - length)
    elif length > maxlen:
        print(" 啊啊啊啊啊啊出错了！maxlen最大才是512！！！现在length居然等于", length)
        print("再cut一次！！！")
        tokens = tokens[:maxlen]

    return tokens


# 将文本截取至maxlen-2的长度
def textsCut(input_texts, maxlen):
    result = []
    print(" maxlen - 2 = ", maxlen - 2)
    for text in input_texts:
        # print("text in textsCut = ", list(text))
        length = len(text)
        # print("length before = ", length)
        if length <= maxlen - 2:
            result.append(text)
            continue
        # print("length after = ", len(text[:maxlen - 2]))
        result.append(text[:maxlen - 2])

    return result


# 创建bert模型
def createBertEmbeddingModel():
    print(">>>开始加载Bert模型。。。")
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
def getBertEmbeddings(bert_model, tokenizer, origin_data, maxlen, debug=False):
    print(">>>正在飞速获取bert字符级向量和句子级向量")
    character_embeddings = []
    sentence_embeddings = []

    input_texts = origin_data["content"]
    input_texts = textsCut(input_texts, maxlen)  # 对长句子进行截断

    for text in input_texts:
        current_embedding = []
        # print("text = ", text)
        # print("text's length = ", len(text))
        tokens = tokenizer.tokenize(text)
        if len(tokens) > maxlen:
            print("tokens = ", tokens)
            print("tokens' length = ", len(tokens))
        tokens = textsPadding(tokens, maxlen)
        # print("tokens after = ", tokens)
        # print("tokens' length = ", len(tokens))
        indices, segments = tokenizer.encode(first=text, max_len=512)
        # print("indices = ", indices[:10])
        # print("segments = ", segments[:10])

        predicts = bert_model.predict([np.array([indices]), np.array([segments])])[0]
        # print("tokens' length = ", len(tokens))
        for i, token in enumerate(tokens):
            # 此处为了限制评论最长是512个字符，后续可以做实验扩大
            # if i >= maxlen:
            #     break

            if debug:
                predicted = predicts[i].tolist()[:5]
            else:
                predicted = predicts[i].tolist()
            # print(token, predicted)
            # 注意此处！！将一个句子的所有字符向量进行拼接extend，方面后面存放
            current_embedding.extend(predicted)
            # current_embedding.append(predicts[i].tolist())
            if token == "[CLS]":
                sentence_embeddings.append(predicted)
        character_embeddings.append(current_embedding)
    # print("character_embeddings[0] = ", character_embeddings[0])
    # print("sentence_embeddings[0] = ", sentence_embeddings[0])

    # print(">>>bert字符级向量和句子级向量GET。。。")

    character_path = config.character_embeddings_validation
    sentence_path = config.sentence_embeddings_validation
    saveCharacterEmbeddings(character_embeddings, character_path)
    saveSentenceEmbeddings(sentence_embeddings, sentence_path)

    return character_embeddings, sentence_embeddings


# 保存字符向量的结果
def saveCharacterEmbeddings(character_embeddings, save_path):
    print(">>>正在保存字符级向量至文件...")
    character_embeddings = np.array(character_embeddings)
    # print("character_embeddings' shape = ", character_embeddings.shape)
    # print("character_embeddings' dim = ", character_embeddings.ndim)

    with open(save_path, 'w') as file_object:
        np.savetxt(file_object, character_embeddings, fmt='%f', delimiter=',')


# 读取字符级向量
def getCharacterEmbeddings(path, debug):
    result = np.loadtxt(path, delimiter=',')
    if debug:
        result = np.reshape(result, (-1, 512, 5))  # 测试
    else:
        result = np.reshape(result, (-1, 512, 768))

    print("character_embeddings' shape = ", result.shape)
    print("character_embeddings' dim = ", result.ndim)

    return result


# 保存句子向量的结果
def saveSentenceEmbeddings(sentence_embeddings, save_path):
    print(">>>正在保存句子级向量至文件...")
    sentence_embeddings = np.array(sentence_embeddings)
    # print("sentence_embeddings' shape = ", sentence_embeddings.shape)
    # print("sentence_embeddings' dim = ", sentence_embeddings.ndim)

    with open(save_path, 'w') as file_object:
        np.savetxt(file_object, sentence_embeddings, fmt='%f', delimiter=',')


# 读取句子级向量
def getSentenceEmbeddings(path):
    result = np.loadtxt(path, delimiter=',')

    print("sentence_embeddings' shape = ", result.shape)
    print("sentence_embeddings' dim = ", result.ndim)

    return result


# 对输入的评论文本向量（一个向量表示一个句子）进行聚类，得到三个聚类中心，并写入文件
def getClusterCenters(sentence_embeddings, cluster_centers_path):
    print(">>>In getClusterCenters of absa_dataProcess.py...")
    kMeans_model = trainKMeansModel(sentence_embeddings, 3)

    # 查看预测样本的中心点
    clusters_centers = kMeans_model.cluster_centers_
    # print("clusters' centers:", clusters_centers)

    # 将聚类中心点写入文件
    with codecs.open(cluster_centers_path, "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(clusters_centers)
        f.close()

    print(" End of getClusterCenters in absa_dataProcess.py...")

    return clusters_centers


# 从bert_model读取情感词向量，然后计算得到聚类中心
def getClusterCentersV2(sentiment_words_path, cluster_centers_path, bert_path, debug):
    # 从情感词典中读取情感词，分三类
    negative_words, neutral_words, positive_words = getSentimentWords(sentiment_words_path, debug)
    # print("negative_words = ", negative_words)
    # print("neutral_words = ", neutral_words)
    # print("positive_words = ", positive_words)

    # 从bert_model中读取词向量
    negative_words_embeddings, neutral_words_embeddings, positive_words_embeddings = getSentimentWordsEmbeddings(negative_words, neutral_words, positive_words, bert_path, debug)
    print("negative_words_embeddings' length = ", len(negative_words_embeddings))
    print("neutral_words_embeddings[0]'s length = ", len(neutral_words_embeddings[0]))
    # print("neutral_words_embeddings = ", neutral_words_embeddings)

    # 计算词向量的聚类中心，保存至文件
    cluster_centers = calculateClusterCenters1(negative_words_embeddings, neutral_words_embeddings, positive_words_embeddings, cluster_centers_path)
    # cluster_centers = calculateClusterCenters2(negative_words_embeddings, neutral_words_embeddings, positive_words_embeddings, cluster_centers_path)

    return cluster_centers


# 读取情感词
def getSentimentWords(path, debug=False):
    negative_words, neutral_words, positive_words = [], [], []

    wb = xlrd.open_workbook(path)
    sheet = wb.sheet_by_name('Sheet1')
    for a in range(sheet.nrows):
        if a <= 0:
            continue
        line = sheet.row_values(a)
        word = str(line[0]).strip()
        polar = line[6]

        if polar == 0:
            neutral_words.append(word)
        elif polar == 1:
            positive_words.append(word)
        else:
            negative_words.append(word)
        if a == 50 and debug:
            break

    return negative_words, neutral_words, positive_words


# 从bert_model中读取词向量
def getSentimentWordsEmbeddings(negative_words, neutral_words, positive_words, bert_path, debug):
    negative_words_embeddings = getSentimentWordsEmbeddingsByList(negative_words, bert_path, debug)
    neutral_words_embeddings = getSentimentWordsEmbeddingsByList(neutral_words, bert_path, debug)
    positive_words_embeddings = getSentimentWordsEmbeddingsByList(positive_words, bert_path, debug)

    return negative_words_embeddings, neutral_words_embeddings, positive_words_embeddings


def getSentimentWordsEmbeddingsByList(texts, bert_path, debug=False):
    final_embeddings = []
    embeddings = extract_embeddings(bert_path, texts)
    for embedding in embeddings:
        # 使用词语中第一个字符向量CLS来表示当前词向量
        # test
        if debug:
            CLS = embedding[0][:5]
        else:
            CLS = embedding[0]
        final_embeddings.append(CLS)

    return final_embeddings


# 计算词向量的聚类中心1，各自聚类一个中心，然后输出三个中心
def calculateClusterCenters1(negative_words_embeddings, neutral_words_embeddings, positive_words_embeddings, cluster_centers_path):

    kMeans_model_negative = trainKMeansModel(negative_words_embeddings, 1)
    # 查看预测样本的中心点
    clusters_centers_negative = kMeans_model_negative.cluster_centers_
    # print("clusters_centers_negative's centers:", clusters_centers_negative)

    kMeans_model_neutral = trainKMeansModel(neutral_words_embeddings, 1)
    # 查看预测样本的中心点
    clusters_centers_neutral = kMeans_model_neutral.cluster_centers_
    # print("clusters_centers_negative's centers:", clusters_centers_neutral)

    kMeans_model_positive = trainKMeansModel(positive_words_embeddings, 1)
    # 查看预测样本的中心点
    clusters_centers_positive = kMeans_model_positive.cluster_centers_
    # print("clusters_centers_negative's centers:", clusters_centers_positive)

    clusters_centers = [clusters_centers_negative, clusters_centers_neutral, clusters_centers_positive]
    # print("clusters_centers = ", clusters_centers)

    # 将聚类中心点写入文件
    with codecs.open(cluster_centers_path, "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(clusters_centers)
        f.close()

    print(" End of calculateClusterCenters1 in absa_dataProcess.py...")

    return clusters_centers


# 计算词向量的聚类中心2，所有词向量一起聚类得到3个聚类中心
def calculateClusterCenters2(negative_words_embeddings, neutral_words_embeddings, positive_words_embeddings, cluster_centers_path):
    negative_words_embeddings.extend(neutral_words_embeddings)
    negative_words_embeddings.extend(positive_words_embeddings)

    kMeans_model = trainKMeansModel(negative_words_embeddings, 3)
    # 查看预测样本的中心点
    clusters_centers = kMeans_model.cluster_centers_
    print("clusters_centers = ", clusters_centers)

    # 将聚类中心点写入文件
    with codecs.open(cluster_centers_path, "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(clusters_centers)
        f.close()

    print(" End of calculateClusterCenters2 in absa_dataProcess.py...")

    return clusters_centers


# 从文件中读取聚类中心向量
def getClusterCenterFromFile(path):
    cluster_center = pd.read_csv(path, header=None).values.tolist()
    # print(">>>cluster_center = ", cluster_center)

    return cluster_center


# 接收sentence_embeddings，训练模型并返回
def trainKMeansModel(sentence_embeddings, n_clusters):
    # 创建KMeans 对象
    cluster = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=-1)
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

    print(">>>评论文本的隶属度计算结束。。。")
    return membership_degrees


# 计算评论中的字向量与聚类中心的隶属度（余弦距离），并存入文件
# character_embeddings太大，所以需要一边读 一边计算隶属度 然后保存
# cluster_centers_path直接读进内存
# 每500条评论就写入一次文件
def calculateAndSaveMembershipDegree(cluster_centers, character_embeddings_train_path, membership_degree_train_path, debug=False):
    i = 1

    f = open(character_embeddings_train_path)
    cache = []

    for line in f:
        line = parseLine2(line, debug)  # 一个line表示一条评论

        sentence_membership_degree = calculateMembershipDegreeForSingleSentence(cluster_centers, line)
        # print("sentence_membership_degree = ", sentence_membership_degree)
        cache.append(sentence_membership_degree)

        # if i % 5000 == 0:
        if i % 500 == 0:  # 测试
            # 写入文件
            writeToFile(cache, membership_degree_train_path)
            cache = []
            print("写入文件。。。i = ", i)

        i += 1
    if len(cache) > 0:
        writeToFile(cache, membership_degree_train_path)


def writeToFile(cache, save_path):
    # 评论文本的词嵌入向量转为2D
    cache = np.reshape(cache, (-1, 512 * 3))

    with open(save_path, 'ab') as file_object:
        np.savetxt(file_object, cache, fmt='%f', delimiter=',')


# 计算一条评论语句与聚类中心的隶属度
def calculateMembershipDegreeForSingleSentence(cluster_centers, character_embedding):
    sentence_membership_degree = []
    for character in character_embedding:
        character_membership_degrees = calculateMembershipDegreeForSingleCharacter(cluster_centers, character)
        sentence_membership_degree.append(character_membership_degrees)
    return sentence_membership_degree


# 计算一个字符向量与聚类中心的隶属度
def calculateMembershipDegreeForSingleCharacter(cluster_centers, character):
    character_membership_degrees = []
    for cluster_center in cluster_centers:
        result = calculateCosinValue(character, cluster_center)
        character_membership_degrees.append(result)
    return character_membership_degrees


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


# 读取评论文本的隶属度
def getMembershipDegrees(path):
    result = np.loadtxt(path, delimiter=',')
    result = np.reshape(result, (-1, 512, 3))

    return result


# 将assisted_vector拼接在main_vector后面
def concatenateVector(main_vector, assisted_vector, save_path, debug=False):
    print(">>>main_vector's length = ", len(main_vector))
    # print("assisted_vector = ", assisted_vector)
    print(">>>assisted_vector's length = ", len(assisted_vector))

    length_1 = len(main_vector)
    length_2 = len(assisted_vector)
    if length_1 != length_2:
        print(">>>字向量和隶属值向量长度不一致!!!")
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

        current_final_word_embeddings = np.concatenate((main_vector_current, assisted_vector_current), axis=1).tolist()
        # print("current_final_word_embeddings = ", current_final_word_embeddings)

        # 直接追加写入文件
        saveFinalEmbeddings(current_final_word_embeddings, save_path, debug)


# 将最终的包含隶属度向量的embedding存入文件
def saveFinalEmbeddings(final_word_embeddings, save_path, debug=False):
    # print(">>>正在保存final_word_embeddings向量至文件...")
    final_word_embeddings = np.array(final_word_embeddings)
    if debug:
        final_word_embeddings = np.reshape(final_word_embeddings, (-1, 512 * 8))
    else:
        final_word_embeddings = np.reshape(final_word_embeddings, (-1, 512 * 771))

    with open(save_path, 'ab') as file_object:
        np.savetxt(file_object, final_word_embeddings, fmt='%f', delimiter=',')


# 将最终的包含隶属度的向量存入文件
def saveFinalEmbeddingLittleByLittle(membership_degrees, embeddings_path, save_path, debug):
    i = 0

    f = open(embeddings_path)
    cache = []
    membership_degree_cache = []

    for line in f:
        line = parseLine2(line, debug)
        cache.append(line)
        membership_degree_cache.append(membership_degrees[i])

        if i % 300 == 0:
            # 写入文件
            concatenateVector(cache, membership_degree_cache, save_path, debug)
            cache = []
            membership_degree_cache = []
            print("final_embeddings写入文件中。。。i = ", i)

        i += 1

    if len(cache) > 0:
        concatenateVector(cache, membership_degree_cache, save_path, debug)


def getFinalEmbeddings(path, debug=False):
    print("正在获取final_word_embeddings。。。")

    result = np.loadtxt(path, delimiter=',')

    if debug:
        result = np.reshape(result, (-1, 512, 8))  # 测试
    else:
        result = np.reshape(result, (-1, 512, 771))

    print("正在获取final_word_embeddings' shape = ", result.shape)
    print("正在获取final_word_embeddings' dim = ", result.ndim)

    return result


# 使用generator yield批量训练数据
def generateTrainSet(X_train, Y_train, batch_size):
    print("X_train's length = ", len(X_train))
    print("Y_train's length = ", len(Y_train))
    for i in range(0, len(X_train), batch_size):
        x = X_train[i: i + batch_size]
        y = Y_train[i: i + batch_size]
        yield np.array(x), to_categorical(y)


# 使用generator yield批量训练数据，从文件中读取X
def generateTrainSetFromFile(X_path, Y_train, batch_size, debug):
    # print("从", X_path, "中读取X_train数据")
    length = len(Y_train)
    # print("Y_train's length = ", length)
    while True:
        f = open(X_path)
        cnt = 0
        X = []
        Y = []
        i = 0  # 记录Y_train的遍历
        cnt_Y = 0
        for line in f:
            X.append(parseLine(line, debug))
            i += 1
            cnt += 1
            if cnt == batch_size or i == length:

                Y = Y_train[cnt_Y: i]
                cnt_Y += batch_size

                cnt = 0
                # print("X's length = ", len(X))
                # print("Y's length = ", len(Y))
                yield (np.array(X), to_categorical(Y))
                X = []
                Y = []


# 使用generator yield生成批量数据，从文件中读取X，只生成X
def generateXFromFile(X_path, y_length, batch_size, debug):
    while True:
        # print("in generateXFromFile...")
        f = open(X_path)
        cnt = 0
        X = []
        i = 0
        for line in f:
            X.append(parseLine(line, debug))
            i += 1
            cnt += 1
            if cnt == batch_size or i == y_length:
                cnt = 0
                # print("X = ", X)
                yield (np.array(X))
                X = []


def parseLine(line, debug=False):
    # print("line = ", line)
    line = [float(x) for x in line.split(',')]
    # print("line = ", line)
    # print("line's length = ", len(line))
    # reshape
    if debug:
        line = np.reshape(list(line), (-1, 8))  # 测试
    else:
        line = np.reshape(line, (-1, 771))
    return line


def parseLine2(line, debug=False):
    # print("line = ", line)
    line = [float(x) for x in line.split(',')]
    # print("line = ", line)
    # print("line's length = ", len(line))
    # reshape
    if debug:
        line = np.reshape(list(line), (-1, 5))  # 测试
    else:
        line = np.reshape(line, (-1, 768))
    return line


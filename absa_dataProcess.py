#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from collections import Counter
import pandas as pd

import numpy as np
import codecs

from sklearn.cluster import KMeans

from keras_bert import Tokenizer as bert_Tokenizer, load_trained_model_from_checkpoint

import absa_config as config


# 读取数据
def initData(debug=False, clean_enter=False, clean_space=False):
    # print("In initData function of dataProcess.py...")
    data = pd.read_csv(config.meituan_validation_new)
    # data = pd.read_csv(config.meituan_train)
    if debug:
        data = data[:50]

    data = data[:1000]

    # 对原评论文本进行清洗，去回车符 去空格
    # print("data['content']_0 = ", data['content'])
    if clean_enter:
        data = dataCleanEnter(data)
    if clean_space:
        data = dataCleanSpace(data)
    # print("data['content']_1 = ", data['content'])

    global y_cols
    # y_cols = data.columns.values.tolist()[2:22]
    y_cols = data.columns.values.tolist()[2:7]
    print(">>>data = ", data.head())
    print(">>>data'type = ", type(data))
    print(">>>data's shape = ", data.shape)
    print(">>>y_cols = ", y_cols)

    # print("end of initData function in dataProcess.py...")

    return data, y_cols


# 对原评论文本进行清洗,去回车符
def dataCleanEnter(data):
    rows = len(data)

    for i in range(rows):
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
    if length == maxlen:
        return tokens
    elif length > maxlen:
        print(" 啊啊啊啊啊啊出错了！maxlen最大才是512！！！现在length居然等于", length)
    pad = '[PAD]'
    tokens += [pad] * (maxlen - length)

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
        # print("tokens = ", tokens)
        # print("tokens' length = ", len(tokens))
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
    kMeans_model = trainKMeansModel(sentence_embeddings)

    # 查看预测样本的中心点
    clusters_centers = kMeans_model.cluster_centers_
    # print("clusters' centers:", clusters_centers)

    # 将聚类中心点写入文件
    with codecs.open(config.cluster_center, "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(clusters_centers)
        f.close()

    print(" End of getClusterCenters in absa_dataProcess.py...")

    return clusters_centers


# 从文件中读取聚类中心向量
def getClusterCenterFromFile():
    cluster_center = pd.read_csv(config.cluster_center, header=None).values.tolist()
    print(">>>cluster_center = ", cluster_center)

    return cluster_center


# 接收sentence_embeddings，训练模型并返回
def trainKMeansModel(sentence_embeddings):
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
    print(">>>main_vector's length = ", len(main_vector))
    # print("assisted_vector = ", assisted_vector)
    print(">>>assisted_vector's length = ", len(assisted_vector))

    final_word_embeddings = []
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

        final_word_embeddings.append(np.concatenate((main_vector_current, assisted_vector_current), axis=1).tolist())
        # print("final_word_embeddings = ", final_word_embeddings)

    return np.array(final_word_embeddings)


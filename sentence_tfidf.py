#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
import jieba
import pandas as pd
from string import digits
import numpy as np

jieba.load_userdict("config/filedWords")

abstract_file_debug = "result/farmer/complaints/abstracts_debug.csv"
abstract_file = "result/farmer/complaints/abstracts_normal_all.csv"

debug = False


# 读取停用词
def getStopwords(path):
    stoplist = []
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        stoplist.append(line.strip())

    return stoplist


# 数据预处理
def dataPreprocessing(texts):
    result = []

    for text in texts:
        # print("text = ", text)
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(text))
        text = text.translate(str.maketrans('', '', digits))
        text = jieba.cut(text)
        text = [word.strip() for word in text if word not in stoplist]
        if len(text) < 1:
            continue
        result.extend(text)
    result = ' '.join(result)

    return result


# 转为词频矩阵
def transMatrix(corpus):
    print("corpus = ", corpus)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()

    # print("word = ", word)
    # print(X.toarray())

    transformer = TfidfTransformer()

    # print("transformer = ", transformer)
    tfidf = transformer.fit_transform(X)
    # print(tfidf.toarray())

    words = vectorizer.get_feature_names()
    weight = tfidf.toarray()
    # print("weigth.shape = ", weight.shape)
    # print("tfidf.shape = ", tfidf.shape)
    # print(weight)
    print("weight's length = ", len(weight))
    print("word's length = ", len(word))
    for i in range(len(weight)):
        for j in range(len(word)):
            # if weight[i][j] > 0:
            print(word[j], weight[i][j])
    '''
    '''

    sort = np.argsort(tfidf.toarray(), axis=1)
    # print("sort.shape = ", sort.shape)
    # top10 = sort[:, -topK:]
    # print("top10", top10)

    key_words = pd.Index(words)[sort].tolist()
    # print(key_words[:10])

    return key_words


# 展示结果
def showResult(keyWords, topK):
    # print("》》》打印每个topic的topK个关键词")
    length = len(keyWords)
    for i in range(length):
        print("topic", str(i), ":", keyWords[i][-topK:])


if __name__ == "__main__":
    print("》》》start at absa_topic_tfidf.py。。。")
    if debug:
        path = abstract_file_debug
    else:
        path = abstract_file

    # 读取摘要csv文件
    data = pd.read_csv(path, header=None, names=['col', 'score'], delimiter=',')

    print("data = ", data.head())
    stoplist = getStopwords('config/stopwords_farmer.txt')

    # print(data['col'])
    print("*" * 100)
    data = data['col']
    # print("data2 = ", data)
    texts = [dataPreprocessing(data.tolist())]
    # print("texts = ", texts)

    # 根据tf-idf计算得到每个topic的主题词
    keyWords = transMatrix(texts)

    # 展示结果
    topK = 500
    showResult(keyWords, topK)

    print("》》》end of absa_topic_tfidf.py。。。")


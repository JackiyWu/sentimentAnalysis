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


# 读取文本文件，文件是字典形式
def readText(path):
    file = open(path, 'r')
    js = file.read()
    texts = json.loads(js)
    # print(texts)
    # print("texts' type = ", type(texts))

    file.close()

    return texts


# 将字典内容转为topic：list形式
def processTexts(texts):
    result = {}

    for topic, content in texts.items():
        current = []
        # print("content[0] = ", content[0])
        for item in content:
            # print("item = ", item)
            for key, value in item.items():
                # print("value = ", value)
                current.append(value)
                # print(value)
        result[topic] = current
        # print("current's length = ", len(current))

    # print("result = ", result)

    return result


# 数据预处理，将每个topic的所有文本合并
def dataPreprocessing2(texts):
    # print(">>>数据预处理中2.。。")
    result = []
    # 去停用词
    # 增加词库
    for topic, lines in texts.items():
        current = []
        for line in lines:
            # 去标点符号
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
            # 去掉数字
            line = line.translate(str.maketrans('', '', digits))
            # 分词
            line = jieba.cut(line)
            line = [word.strip() for word in line if word not in stoplist]
            if len(line) < 1:
                continue
            current.extend(line)
        current = ' '.join(current)
        # print(topic, current)

        result.append(current)

    return result


# 数据预处理，分词、去停用词
def dataPreprocessing(texts):
    print(">>>数据预处理中。。。")
    result = {}
    # 去停用词
    stoplist = getStopList()
    # 增加词库
    jieba.load_userdict("config/filedWords")
    for topic, lines in texts.items():
        current = []
        for line in lines:
            # 去标点符号
            line = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(line))
            # 去掉数字
            line = line.translate(str.maketrans('', '', digits))
            # 分词
            line = jieba.cut(line)
            line = [word.strip() for word in line if word not in stoplist]
            if len(line) < 1:
                continue
            line = ' '.join(line)
            current.append(line)
        # print(topic, ":", current)

        result[topic] = current

    return result


def getStopList():
    stoplist = pd.read_csv('config/stopwords.txt').values
    return stoplist


stoplist = getStopList()


# 转为词频矩阵
def transMatrix(corpus):
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
    '''
    for i in range(len(weight)):
        for j in range(len(word)):
            if weight[i][j] > 0:
                print(word[j], weight[i][j])
    '''

    sort = np.argsort(tfidf.toarray(), axis=1)
    # print("sort.shape = ", sort.shape)
    # top10 = sort[:, -topK:]
    # print("top10", top10)

    key_words = pd.Index(words)[sort].tolist()
    # print(key_words[0])

    return key_words


# 展示结果
def showResult(keyWords, topK):
    # print("》》》打印每个topic的前10个关键词")
    length = len(keyWords)
    for i in range(length):
        print("topic", str(i), ":", keyWords[i][-topK:])


if __name__ == "__main__":
    print("》》》start at absa_topic_tfidf.py。。。")

    names = ["kuaileai"]

    topK = 15
    for restaurant_name in names:
        print("》》》current restaurant is ", restaurant_name)
        for number in range(4, 46):
            print("》》》current topics is ", number)
            # 读取文本文件
            path = "result/pred_text/KMeans_" + str(number) + "_" + restaurant_name + ".txt"
            texts = readText(path)

            # 将字典内容转为topic：list形式
            texts = processTexts(texts)
            # print("texts = ", texts)

            # 数据预处理，分词、去停用词
            texts = dataPreprocessing2(texts)
            print("texts = ", texts)

            # 根据tf-idf计算得到每个topic的主题词
            keyWords = transMatrix(texts)

            # 展示结果
            showResult(keyWords, topK)

    print("》》》end of absa_topic_tfidf.py。。。")


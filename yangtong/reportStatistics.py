#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
1.所有变量使用下划线命名，data_source
2.所有函数使用驼峰命名法，getDataSource()
'''

from yangtong.reportStatistics_tools import *

if __name__ == "__main__":
    print(">>>begin in reportStatistics.py ...")
    # hello()

    # 1. 获取数据源
    print("********************获取数据源********************")
    # file_dir = "C:\Softwares\coding\workspace\sentimentAnalysis\yangtong\\files"
    file_dir = "C:\\Users\\wujie\Desktop\\Backup of laboratory-202101\\Things\Yang\\text mining from goverment reports\\files"
    data_source = getDataSource(file_dir)
    # data_source = getTestDataSource()

    # 2. 计算热词/新词
    print("********************计算热词********************")
    limit = 50
    results, words, weights = computeHotWords(data_source, limit)
    print("results = ", results)
    print("********************热词计算结束********************")

    print("********************计算新词********************")
    # new_words = computeNewAndInheritWords(data_source, limit, word, weights)
    # print("new_words = ", new_words)
    print("********************新词计算结束********************")

    '''
    # 3. 单词按照特征（在文档中出现频率）聚类
    data_source_by_document = getDataSourceByDocument(file_dir)
    print("data_source_by_document = ", data_source_by_document)
    # 计算词频
    word_frequency, words = computeWordFrequency(data_source_by_document)
    # print(word_frequency.shape)
    '''
    # 聚类
    n_cluster = 10
    # KMeans_result = trainKMeansModel(words, word_frequency, n_cluster)
    # 使用tf-idf聚类
    print("weights'lenght = ", len(weights))
    print("weights'type = ", type(weights))
    KMeans_result_tfidf = trainKMeansModel(words, weights, n_cluster)

    # 4. 保存结果
    print("********************保存结果********************")
    file_path = "C:\\Users\\wujie\Desktop\\Backup of laboratory-202101\\Things\Yang\\text mining from goverment reports\\result\\word_frequency.csv"
    # saveToFile(word_frequency, file_path)

    print(">>>end at reportStatistics.py ...")

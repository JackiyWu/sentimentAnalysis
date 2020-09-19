#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import numpy as np
import xlrd
import fuzzySentiment.config as config


# 词语-序号字典,脏乱:0
word_index = {}
# 序号-词语字典，0:脏乱
index_word = {}
# 词语的特征,脏乱:[category, intensity, polar]
word_feature = {}
# 情感类别特征转为数字,共21个类别
sentiment_category = {'PA': 1, 'PE': 2, 'PD': 3, 'PH': 4, 'PG': 5, 'PB': 6, 'PK': 7, 'NA': 8, 'NB': 9, 'NJ': 10, 'NH': 11, 'PF': 12, 'NI': 13, 'NC': 14, 'NG': 15, 'NE': 16, 'ND': 17, 'NN': 18, 'NK': 19, 'NL': 20, 'PC': 21}
# 聚类中心 2-中性，3-正向，1-负向
clusters_centers = [[0.99347826, 0.98695652, 2.], [0.38799172, 1.06376812, 3.], [1.67957958, 1.10690691, 1.01501502]]


#读取excel文件
def read_excel():
    print(">>>in read_excel function in KMeansCluster.py...")
    wb = xlrd.open_workbook(config.sentiment_dictionary) # 打开Excel文件
    sheet = wb.sheet_by_name('Sheet1') #通过excel表格名称(rank)获取工作表
    dat = [] #创建空list
    index = 0
    for a in range(sheet.nrows): #循环读取表格内容（每次读取一行数据）
        if a <= 0:
            continue
        line = sheet.row_values(a)
        # print(line)
        word = str(line[0]).strip()
        category = sentiment_category[line[4].strip()] / 10.0
        intensity = line[5] / 5.0
        # 需要把情感词汇本体库中的词汇极性转换成跟输入语料的标注一致
        if line[6] == 0:
            polar = 2.0  # 中性
        elif line[6] == 1:
            polar = 3.0  # 正向
        else:
            polar = 1.0  # 负向
        # polar = line[6]

        # res = [intensity, polar]
        res = [category, intensity, polar]
        dat.append(res)

        # 存入字典
        word_index[word] = index
        index_word[index] = word
        if word not in word_feature.keys():
            word_feature[word] = res
        index += 1
        # print(index, word, polar)

        # if index >= 1000:
        #     break

    print(">>>end of read_excel function in KMeansCluster.py...")

    return word_feature


# 读取本体库，不做修改
def read_excel2():
    print("in read_excel2 function in KMeansCLuster.py...")
    wb = xlrd.open_workbook(config.sentiment_dictionary)
    sheet = wb.sheet_by_name('Sheet1')
    dat = []
    index = 0
    for a in range(sheet.nrows):
        if a <= 0:
            continue
        line = sheet.row_values(a)
        # print(line)
        word = str(line[0]).strip()
        intensity = line[5]
        # 需要把情感词汇本体库中的词汇极性转换成跟输入语料的标注一致
        if line[6] == 0:
            polar = 2.0  # 中性
        elif line[6] == 1:
            polar = 3.0  # 正向
        else:
            polar = 1.0  # 负向

        # res = [intensity, polar]
        res = [intensity, polar]
        dat.append(res)

        # 存入字典
        word_index[word] = index
        index_word[index] = word
        if word not in word_feature.keys():
            word_feature[word] = res
        index += 1
        # print(index, word, polar, intensity)

        # if index >= 100:
        #     break

    print(">>>end of read_excel function in KMeansCluster.py...")

    return word_feature


# 接收情感词特征数据，训练模型并返回
def train_model(input):
    print(">>>in train_model of KMeansCluster.py...")

    # 创建KMeans 对象
    cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)
    print("input.values() = ", list(input.values()))
    result = cluster.fit(list(input.values()))

    print(">>>end of train_model in KMeansCluster.py...")

    return result


# 查找某个词的特征
def get_feature(word, word_feature):
    feature = word_feature.get(word, [])
    return np.array(feature)


def main():
    features, word_index, index_word = read_excel()
    print(features)

    model = train_model(features)

    # 查看预测的类
    # print(model.labels_)
    cluster_0 = []
    cluster_1 = []
    cluster_2 = []
    for index in range(len(model.labels_)):
        label = model.labels_[index]
        word = index_word.get(index)
        if label == 0:
            cluster_0.append(word)
        elif label == 1:
            cluster_1.append(word)
        else:
            cluster_2.append(word)
    print("cluster_0:", cluster_0[:100])
    print("cluster_1:", cluster_1[:100])
    print("cluster_2:", cluster_2[:100])
    print("*" * 500)

    # 预测
    # print("predict:", model.predict(new_observation))
    # 查看预测样本的中心点
    clusters_centers = model.cluster_centers_
    print("clusters' centers:", clusters_centers)
    '''
    1-褒义，2-贬义，0-中性
    [[3.87306557e-01 1.09913230e+00 1.00500939e+00]
     [1.66388380e+00 1.07498167e+00 1.99789223e+00]
     [9.79479070e-01 9.57804651e-01 9.39248679e-14]]
    '''


if __name__ == "__main__":
    print(">>>KMeansCluster starts working...")

    # features = read_excel()
    # features = np.array(features)
    # print(features)
    print(clusters_centers)
    clusters_centers = np.array(clusters_centers)
    print(clusters_centers)

    # main()

    print(">>>end of KMeansCluster.py...")


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import xlrd
from collections import Counter

import config
import KMeansCluster
import dataProcess


# 计算语料库中每个input包含情感词的个数
# 查询每个input包含情感词的特征向量
def calculate_sentiment_words_feature(input, word_feature):
    print(">>>in calculate_sentiment_words function in fuzzySystem.py...")

    '''
    # 首先拿到情感词汇本体库
    word_feature, word_index, index_word = KMeansCluster.read_excel()
    print(len(word_feature))
    '''

    # 输入语料的情感特征向量
    input_word_feature = []
    for text in input:
        wf = []
        for word in text:
            if word in word_feature.keys():
                wf.append(word_feature.get(word))
                # print("word = ", word)
        # print("text = ", text, "wf = ", wf)
        # 如果输入语料中不包含情感词汇，会传入一个空的列表
        input_word_feature.append(wf)
    # print("input_word_feature:", input_word_feature)

    print(">>>end of calculate_sentiment_words function in fuzzySystem.py...")

    return input_word_feature


# 计算某个数据属于某个类别的隶属度
def cal_fuzzy_membership_degree(features, clusters_centers, texts, ratios):
    print(">>>calculate membership degree in fuzzySystem.py...")
    print("features.type = ", type(features))
    print("clusters_centers.type = ", type(clusters_centers))
    # 计算样本点到所有中心点的距离
    distance = []
    # print("texts = ", texts)
    i = 0
    for feature in features:
        d = []
        # print("feature = ", feature)
        i += 1

        for f in feature:
            d.append(cal_distance(f, clusters_centers))
        d = np.array(d)
        distance.append(d)

    # print("distance = ", distance)
    # distance = np.array(distance)
    # print("distance's type = ", type(distance))
    # 计算样本的隶属度
    membership_degree = membership_degree_solo(distance)

    membership_degree_fuzzy = fuzzy_calculate(membership_degree)

    print("end of fuzzySystem function in fuzzySystem.py...")

    # 数据划分，重新分为训练集，测试集和验证集
    membership_degrer_fuzzy_length = membership_degree_fuzzy.shape[0]
    print("membership_degree_fuzzy_length = ", membership_degrer_fuzzy_length)
    size_train = int(membership_degrer_fuzzy_length * ratios[0])
    size_test = int(membership_degrer_fuzzy_length * ratios[1])

    dealed_train = membership_degree_fuzzy[: size_train]
    dealed_val = membership_degree_fuzzy[size_train: (size_train + size_test)]
    dealed_test = membership_degree_fuzzy[(size_train + size_test):]

    return dealed_train, dealed_val, dealed_test


# 根据输入语料的极性和强度特征，计算其模糊特征
def calculate_fuzzy_feature(features, ratios):
    print(">>>in calculate_fuzzy_feature function of fuzzySystem.py...")
    # print("feature2 = ", features)
    # fuzzy_feature分别保存负向、中性、正向的最终特征值
    fuzzy_feature = []
    # 对features中的每一条输入语料求均值
    features_mean = []
    for feature in features:
        # print("feature_0 = ", feature)
        if len(feature) > 0:
            features_mean.append(np.mean(feature, axis=0))
        else:
            features_mean.append([0, 0])

    # 遍历输入语料的均值，计算其对应的模糊特征
    for feature in features_mean:
        # print("feature1 = ", feature)
        fuzzy_feature.append(calculate_fuzzy_feature_by_polar_intensity(feature))

    fuzzy_feature = np.array(fuzzy_feature)

    # 数据划分，重新分为训练集，测试集和验证集
    membership_degrer_fuzzy_length = fuzzy_feature.shape[0]
    print("membership_degree_fuzzy_length = ", membership_degrer_fuzzy_length)
    size_train = int(membership_degrer_fuzzy_length * ratios[0])
    size_test = int(membership_degrer_fuzzy_length * ratios[1])

    dealed_train = fuzzy_feature[: size_train]
    dealed_val = fuzzy_feature[size_train: (size_train + size_test)]
    dealed_test = fuzzy_feature[(size_train + size_test):]

    print("end of fuzzySystem function in fuzzySystem.py...")

    return dealed_train, dealed_val, dealed_test


# 根据传入的polar和intensity计算其模糊特征
def calculate_fuzzy_feature_by_polar_intensity(feature):
    # print("feature = ", feature)
    result = []
    # print("feature = ", feature)
    feature = list(feature)
    if len(feature) <= 0:
        return result
    intensity = feature[0]
    polar = feature[1]

    # 计算intensity的隶属度
    intensity_membership = calculate_intensity_membership_degree(intensity)
    # print("intensity_membership = ", intensity_membership)
    # 计算polar的隶属度
    polar_membership = calculate_polar_membership_degree(polar)
    # print("polar_membership = ", polar_membership)

    # 计算负向、中性、正向三种特征值
    Negative, Neural, Positive = calculate_feature(polar_membership, intensity_membership)
    if polar == 0 and intensity == 0:
        # print("polar intensity 都是0")
        Negative, Neural, Positive = 0, 0, 0
    result = [Negative, Neural, Positive]
    # print("result = ", result)

    return result


def calculate_feature(polar_membership, intensity_membership):
    polar_membership_1 = polar_membership[0]
    polar_membership_2 = polar_membership[1]
    polar_membership_3 = polar_membership[2]

    intensity_membership_1 = intensity_membership[0]
    intensity_membership_2 = intensity_membership[1]
    intensity_membership_3 = intensity_membership[2]

    Neural = [min(polar_membership_1, intensity_membership_1), min(polar_membership_2, intensity_membership_1),
              min(polar_membership_3, intensity_membership_1), min(polar_membership_2, intensity_membership_2),
              min(polar_membership_2, intensity_membership_3)]

    Negative = [min(polar_membership_1, intensity_membership_3), min(polar_membership_1, intensity_membership_2)]

    Positive = [min(polar_membership_3, intensity_membership_3), min(polar_membership_3, intensity_membership_2)]

    # print("Neural = ", Neural)
    # print("Negative = ", Negative)
    # print("Positive = ", Positive)

    Neural = np.sum(Neural) / count_number(Neural) if count_number(Neural) > 0 else 0
    Negative = np.sum(Negative) / count_number(Negative) if count_number(Negative) > 0 else 0
    Positive = np.sum(Positive) / count_number(Positive) if count_number(Positive) > 0 else 0

    # print("Neural = ", Neural)
    # print("Negative = ", Negative)
    # print("Positive = ", Positive)

    return Negative, Neural, Positive


# 统计列表的非零值数量
def count_number(ll):
    number = 0
    for l in ll:
        if l != 0:
            number += 1

    return number


# 计算polar的隶属度
def calculate_polar_membership_degree(polar):
    # result的三个值一次是负向、中性、正向的隶属度
    result = []

    # 计算负向隶属度
    if polar < 0.75:
        result.append(1)
    elif polar > 1.5:
        result.append(0)
    else:
        result.append(-4 * polar / 3 + 2)

    # 计算中性隶属度
    if polar < 0.75 or polar > 2.25:
        result.append(0)
    elif 0.75 <= polar <= 1.5:
        result.append(4 * polar / 3 - 1)
    else:
        result.append(-4.0 * polar / 3 + 3)

    # 计算正向隶属度
    if polar > 2.25:
        result.append(1)
    elif polar < 1.5:
        result.append(0)
    else:
        result.append(4 * polar / 3 - 2)

    return result


# 计算intensity的隶属度
def calculate_intensity_membership_degree(intensity):
    # result的三个值依次是弱、中、强的隶属度
    result = []

    # 计算弱的隶属度
    if intensity <= 2.5:
        result.append(1)
    elif intensity > 5:
        result.append(0)
    else:
        result.append(-0.4 * intensity + 2)

    # 计算中 的隶属度
    if intensity < 2.5 or intensity > 7.5:
        result.append(0)
    elif 2.5 < intensity < 5:
        result.append(0.4 * intensity - 1)
    else:
        result.append(-0.4 * intensity + 3)

    # 计算强 的隶属度
    if intensity < 5:
        result.append(0)
    elif intensity > 7.5:
        result.append(1)
    else:
        result.append(0.4 * intensity - 2)

    return result


# 传入距离矩阵，输出对应的隶属度
def membership_degree_solo(distance):
    membership_degree = []
    # print("distance = ", distance)
    for dis in distance:
        if len(dis) <= 0:
            membership_degree.append([])
            continue
        # print("dis = ", dis)
        membership = []
        dis_turn = 1 / dis  # 距离的倒数
        # print("dis_turn = ", dis_turn)
        dis_sum = dis_turn.sum(axis=1)  # 计算每个词向量的距离和
        for i in range(len(dis_turn)):
            temp = dis_turn[i] / dis_sum[i]
            # print("temp = ", temp)
            membership.append(list(temp))
            # print("membership'type1 = ", type(membership))
        # print("membership = ", membership)
        # for m in membership:
            # print("m's type = ", type(m))
        membership_degree.append(membership)
        # membership_degree.append(np.array(membership))
    # print("membership_degree = ", np.array(membership_degree))

    # return np.array(membership_degree)
    return membership_degree


# 传入总体隶属度，输出3个维度的特征
def fuzzy_calculate(membership_degree):
    print(">>>in fuzzy_calculate function of fuzzySystem.py...")
    result = []
    for input_membership_degree in membership_degree:
        if input_membership_degree:
            # print("input_membershi_degree = ", input_membership_degree)
            result.append(list(np.average(input_membership_degree, axis=0)))
            # result.append(np.average(input_membership_degree, axis=0))
        else:
            result.append([0., 0., 0.])
            # result.append(np.array([0., 0., 0.]))

    # print("result = ", result)

    print(">>>end of fuzzy_calculate function in fuzzySystem.py...")

    return np.array(result)


# 传入情感隶属度特征，得到输入向量的维度
def calculate_input_dimension(input_word_membership_degree):
    return len(input_word_membership_degree[0])


# 定义欧几里得距离
def cal_distance(vector1, centers):
    vector1 = np.array(vector1)

    distance = []
    for center in centers:
        center = np.array(center)
        distance.append(np.sqrt(np.sum(np.square(vector1 - center))))
    # print("distance = ", distance)

    return distance


'''
if __name__ == '__main__':
    print(">>>in fuzzySystem.py...")

    clusters_centers = KMeansCluster.clusters_centers

    print('clusters_centers = ', clusters_centers)

    origin_data = dataProcess.initData()[0]
    print("origin_data's shape = ", origin_data.shape)
    stoplist = dataProcess.getStopList()
    # print(len(origin_data))
    # print(origin_data)
    # print(len(stoplist))

    # 获取输入语料的文本
    input_texts = dataProcess.processDataToTexts(origin_data, stoplist)
    # print("input_texts = ", input_texts)

    # 首先拿到情感词汇本体库
    word_feature, word_index, index_word = KMeansCluster.read_excel()
    print(len(word_feature))

    input_word_feature = calculate_sentiment_words_feature(input_texts, word_feature)
    print(input_word_feature)

    input_word_membership_degree = cal_fuzzy_membership_degree(input_word_feature, clusters_centers, input_texts)
    print("input_word_membership_degree = ", input_word_membership_degree)
    # print("input_word_membership_degree.shape = ", input_word_membership_degree.shape)

    # 统计输入语料的文本包含多少个情感词汇
    sentiment_words_count = []
    for text in input_texts:
        count = 0
        for word in text:
            if word in word_feature.keys():
                count += 1
        sentiment_words_count.append(count)
    # print("sentiment_words_count = ", sentiment_words_count)
    # print("sentiment_words_count'length = ", len(sentiment_words_count))
    # print(Counter(sentiment_words_count))

    print(">>>end of fuzzySystem.py...")
'''


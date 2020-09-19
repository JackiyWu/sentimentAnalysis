#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.preprocessing import sequence


def create_ngram_set(input_list, ngram_value=2):
    """
    从整数列表中提取一组 n 元语法。

    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    通过附加 n-gram 值来增强列表（序列）的输入列表。

    示例：添加 bi-gram
    sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    示例：添加 tri-gram
    sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# 获取增广特征
def get_add_feature(sentences):
    print(">>>in get_add_feature function of fasttext.py...")
    # print("sentences = ", sentences)
    char_set = set(word for sen in sentences for word in sen)
    # print("char_set = ", char_set)
    char_dic = {j: i + 1 for i, j in enumerate(char_set)}
    char_dic["unk"] = 0
    # print("char_dic:", char_dic)
    print("length of char_dic = ", len(char_dic.keys()))
    print("-" * 100)
    # n-gram特征增广
    max_features = len(char_dic)
    sentences2id = [[char_dic.get(word) for word in sen] for sen in sentences]
    # print("sentences2id before = ", sentences2id)
    ngram_range = 2
    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in sentences2id:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
        # Dictionary mapping n-gram token to a unique integer. 将 ngram token 映射到独立整数的词典
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        # 整数大小比 max_features 要大，按顺序排列，以避免与已存在的特征冲突
        start_index = max_features
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        # print("token_indice's length:", len(token_indice.keys()))
        # print("token_indice[0]:", list(token_indice.keys())[0], ":", list(token_indice.values())[0])
        # print("token_indice[2]:", list(token_indice.keys())[2], ":", list(token_indice.values())[2])

    fea_dict = {**token_indice, **char_dic}
    # 使用 n-gram 特征增广 X_train
    # print("sentences2id before = ", sentences2id)
    sentences2id = add_ngram(sentences2id, fea_dict, ngram_range)

    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, sentences2id)), dtype=int)))
    print('Max train sequence length: {}'.format(
        np.max(list(map(len, sentences2id)))))
    print('Min train sequence length: {}'.format(
        np.min(list(map(len, sentences2id)))))
    print("-" * 100)
    # print("sentences2id = ", sentences2id)

    # 增加2gram特征后输入语料太长，需要截断,后续可以考虑取正态分布
    target_length = np.mean(list(map(len, sentences2id)), dtype=int)
    sentences2id_new = []
    for sentence in sentences2id:
        if len(sentence) > target_length:
            sentence = sentence[:target_length]
        sentences2id_new.append(sentence)

    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, sentences2id_new)), dtype=int)))
    print('Max train sequence length: {}'.format(
        np.max(list(map(len, sentences2id_new)))))
    print('Min train sequence length: {}'.format(
        np.min(list(map(len, sentences2id_new)))))
    print("-" * 100)

    print(">>>end of get_add_feature function in fasttext.py...")

    return sentences2id_new, fea_dict


# 处理数据生成训练集 验证集 测试集
def processData(sententce2id, data, maxlen, ratios):
    print(">>>in processData function of fasttext.py..")

    data_T = sequence.pad_sequences(sententce2id, maxlen=maxlen)

    # 数据划分，重新划分为训练集，测试集和验证集
    data_length = data_T.shape[0]
    print("data_length = ", data_length)
    size_train = int(data_length * ratios[0])
    size_test = int(data_length * ratios[1])

    dealed_train = data_T[: size_train]
    dealed_val = data_T[size_train: (size_train + size_test)]
    dealed_test = data_T[(size_train + size_test):]

    train = data[: size_train]
    val = data[size_train: (size_train + size_test)]
    test = data[(size_train + size_test):]

    print(">>>end of processData function in fasttext.py..")

    return dealed_train, dealed_val, dealed_test, train, val, test


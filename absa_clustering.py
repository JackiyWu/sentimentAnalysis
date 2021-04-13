#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import pandas as pd
import codecs
import json

from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from keras_bert import Tokenizer as bert_Tokenizer, load_trained_model_from_checkpoint, extract_embeddings, get_custom_objects
import tensorflow as tf

import absa_models

# restaurant_names = ["dingxiangyuan", "kuaileai"]
# restaurant_names = ["jianshazui", "xiaolongkan", "shouergong", "dingxiangyuan", "jialide", "jiefu", "kuaileai", "niuzhongniu", "zhenghuangqi"]
restaurant_names = ["dingxiangyuan"]


def readAbstract(filepath):
    abstract = []
    f = open(filepath, 'r', encoding='gbk')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line.strip()) < 1:
            continue
        # print("line = ", line)
        temp = line.split(',')
        # print("temp = ", temp)
        abstract.extend(temp)

    return abstract


# 获取Bert词向量
# 获取句子级的向量表示
def getBertEmbeddings(input_texts, debug=False):
    print(">>>正在获取Bert词向量")
    print(">>>构建Bert模型")
    bert_model, tokenizer, token_dict = createBertEmbeddingModel()

    sentence_embeddings = []

    for text in input_texts:
        tokens = tokenizer.tokenize(text)
        indices, segments = tokenizer.encode(first=text, max_len=512)
        predicts = bert_model.predict([np.array([indices]), np.array([segments])])[0]

        for i, token in enumerate(tokens):
            if debug:
                predicted = predicts[i].tolist()[:5]
            else:
                predicted = predicts[i].tolist()

            if token == "[CLS]":
                sentence_embeddings.append(predicted)

    # print("embeddings = ", sentence_embeddings)
    if debug:
        sentence_embeddings = sentence_embeddings[:150]
    return sentence_embeddings


# 创建bert模型
def createBertEmbeddingModel():
    print(">>>开始加载Bert模型。。。")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        token_dict = {}
        bert_dict_path = "config/keras_bert/chinese_L-12_H-768_A-12/vocab.txt"
        with codecs.open(bert_dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        bert_config_path = "config/keras_bert/chinese_L-12_H-768_A-12/bert_config.json"
        bert_checkpoint_path = "config/keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        model = load_trained_model_from_checkpoint(bert_config_path, bert_checkpoint_path)
        tokenizer = bert_Tokenizer(token_dict)

    print(">>>Bert模型加载结束。。。")

    return model, tokenizer, token_dict


# 谱聚类
def spectralClusteringPredict(embeddings, restaurant_name, dim):
    print(">>>开始训练谱聚类模型。。。")

    embeddings = np.array(embeddings)
    cluster_name = 'spectral_' + str(dim)

    # 初步展示一下数据
    plt_name = 'originData'
    # draw2D(X, restaurant_name, plt_name)
    draw3D(embeddings, restaurant_name, plt_name, cluster_name)

    # 使用谱聚类模型默认参数看分类效果
    y_pred = SpectralClustering().fit_predict(embeddings)
    print("Default SpectralClustering's Calinski-Harabasz Score", metrics.calinski_harabasz_score(embeddings, y_pred))
    plt_name = 'defaultSpectral'
    # draw2D(X, y_pred, restaurant_name, plt_name)
    draw3D(embeddings, restaurant_name, plt_name, cluster_name, y_pred)

    # 根据不同参数训练模型，得到最优参数
    k_calinski, gamma_calinski, k_silhouette, gamma_silhouette, k_dbi, gamma_dbi = trainSpectralClustering(embeddings)
    print("k_calinski", k_calinski, ", gamma_calinski = ", gamma_calinski)
    print("k_silhouette", k_silhouette, ", gamma_silhouette = ", gamma_silhouette)
    print("k_dbi", k_dbi, ", gamma_dbi = ", gamma_dbi)

    # 使用最优参数对目标数据聚类
    y_pred = SpectralClustering(n_clusters=k_calinski, gamma=gamma_calinski).fit_predict(embeddings)
    print("Trained SpectralClustering's Calinski-Harabasz Score", metrics.calinski_harabasz_score(embeddings, y_pred))
    plt_name = 'optimizationSpectral'
    # draw2D(X, y_pred, restaurant_name, plt_name)
    draw3D(embeddings, restaurant_name, plt_name, cluster_name, y_pred)
    # plt.show()

    return y_pred


# 训练谱聚类模型
def trainSpectralClustering(X):
    # 训练过程中保存中间的误差值
    k_calinski = 3
    gamma_calinski = 0.01
    best_score_calnski = -1  # 越大越好
    best_score_silhouette = -1  # 越大越好
    best_score_dbi = 2  # 越小越好

    score_calnski = {}  # key为gamma和k
    score_silhouette = {}
    score_dbi = {}

    for index1, gamma in enumerate((0.01, 0.1, 1, 10)):
        # for index2, k in enumerate((3, 4, 5, 6)):
        for k in range(3, 20):
            y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)

            # Calinski-Harabasz的值越大，代表聚类效果越好
            calinski_score = metrics.calinski_harabasz_score(X, y_pred)
            # print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:", calinski_score)
            if calinski_score > best_score_calnski:
                best_score_calnski = calinski_score
                k_calinski = k
                gamma_calinski = gamma

            # Silhouette Coefficient值越大，代表聚类效果越好，取值范围为-1~1之间
            silhouette_score = metrics.silhouette_score(X, y_pred)
            # print("Silhouette Coefficient Score with gamma=", gamma, "n_clusters=", k, "score:", silhouette_score)
            if silhouette_score > best_score_silhouette:
                best_score_silhouette = silhouette_score
                k_silhouette = k
                gamma_silhouette = gamma

            # DBI的值越小，代表聚类效果越好，最小是0
            dbi_score = metrics.davies_bouldin_score(X, y_pred)
            # print("DBI Score with gamma=", gamma, "n_clusters=", k, "score:", dbi_score)
            if dbi_score < best_score_dbi:
                best_score_dbi = dbi_score
                k_dbi = k
                gamma_dbi = gamma

    return k_calinski, gamma_calinski, k_silhouette, gamma_silhouette, k_dbi, gamma_dbi


# KMeans
def KMeansClusteringPredict(X, restaurant_name, cluster_number):
    print(">>>开始训练KMeans聚类模型。。。")
    X = np.array(X)
    cluster_name = "KMeans_clusterNumber_" + str(cluster_number)

    # 初步展示一下数据
    plt_name = 'originData'
    # draw2D(X, restaurant_name, plt_name)
    draw3D(X, restaurant_name, plt_name, cluster_name)

    # 使用KMeans模型默认参数看分类效果
    y_pred = KMeans().fit_predict(X)
    calinski_score = metrics.calinski_harabasz_score(X, y_pred)
    print("KMeansPredict's Calinski-Harabasz Score", calinski_score)
    plt_name = 'defaultKMeans'
    # draw2D(X, y_pred, restaurant_name, plt_name)
    draw3D(X, restaurant_name, plt_name, cluster_name, y_pred)

    '''
    # 根据不同参数训练模型，得到最优参数
    k_calinski, k_silhouette, k_dbi = trainKMeansClustering(X)
    print("k_calinski", k_calinski, ", k_silhouette = ", k_silhouette, ", k_dbi = ", k_dbi)
    '''

    # 使用最优参数对目标数据聚类
    y_pred = KMeans(n_clusters=cluster_number).fit_predict(X)
    print("Trained KMeans' Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, y_pred))
    plt_name = 'optimizationKMeans'
    # draw2D(X, y_pred, restaurant_name, plt_name)
    draw3D(X, restaurant_name, plt_name, cluster_name, y_pred)

    calinski_score = metrics.calinski_harabasz_score(X, y_pred)

    # Silhouette Coefficient值越大，代表聚类效果越好，取值范围为-1~1之间
    silhouette_score = metrics.silhouette_score(X, y_pred)
    print("KMeansPredict Silhouette Score:", silhouette_score)

    # DBI的值越小，代表聚类效果越好，最小是0
    dbi_score = metrics.davies_bouldin_score(X, y_pred)
    print("KMeansPredict DBI Score:", dbi_score)

    # 保存三个评价指标的值
    # print("KMeans: calinski_score = ", calinski_score, ", silhouette_score = ", silhouette_score, ", dbi_score = ", dbi_score)

    return y_pred, calinski_score, silhouette_score, dbi_score


# 画二维图
def draw2D(X, restaurant_name, plt_name, y_pred=None, marker='.'):
    x = X[:, 0]
    y = X[:, 1]
    if y_pred is not None:
        plt.scatter(x, y, c=y_pred, marker=marker)
    else:
        plt.scatter(x, y, marker=marker)
    plt.savefig('result/clustering/2D/figures/' + restaurant_name + '_' + plt_name + '.jpg')

    # 重置画板
    plt.figure()
    plt.close()


# 画三维图
def draw3D(X, restaurant_name, plt_name, cluster_name, y_pred=None):
    label_font = {
        'color': 'b',
        'size': 15,
        'weight': 'bold'
    }

    fig = plt.figure(figsize=(10, 8))   # 参数依然是图片大小
    # fig = plt.figure(figsize=(16, 12))   # 参数依然是图片大小
    ax = fig.add_subplot(111, projection='3d')      # 确定子坐标轴，111表示1行1列的第一个图   要同时画好几个图的时候可以用这个

    # 准备数据
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    if y_pred is not None:
        ax.scatter(x, y, z, c=y_pred, marker='.')   # marker的尺寸和z的大小成正比
        # ax.scatter(x, y, z, c=y_pred, marker='.', s=200)   # marker的尺寸和z的大小成正比
    else:
        ax.scatter(x, y, z, marker='.')   # marker的尺寸和z的大小成正比
        # ax.scatter(x, y, z, marker='.', s=200)   # marker的尺寸和z的大小成正比

    ax.set_xlabel("feature 1", fontdict=label_font)
    ax.set_ylabel("feature 2", fontdict=label_font)
    ax.set_zlabel("feature 3", fontdict=label_font)
    ax.set_title("Clustering visualization", color="b", size=25, weight='bold')   # 子图（其实就一个图）的title
    ax.legend(loc="upper left")   # legend的位置位于左上

    plt.savefig('result/clustering/3D/figures/' + cluster_name + '_' + restaurant_name + '_' + plt_name + '.jpg')

    plt.figure()
    plt.close()


# KMeans训练
def trainKMeansClustering(X):
    # 训练过程中保存中间的误差值
    k_calinski = 3
    best_score_calnski = -1  # 越大越好
    best_score_silhouette = -1  # 越大越好
    best_score_dbi = 2  # 越小越好

    score_calnski = {}  # key为gamma和k
    score_silhouette = {}
    score_dbi = {}

    for k in range(3, 20):
        y_pred = KMeans(n_clusters=k).fit_predict(X)

        # Calinski-Harabasz的值越大，代表聚类效果越好
        calinski_score = metrics.calinski_harabasz_score(X, y_pred)
        # print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:", calinski_score)
        if calinski_score > best_score_calnski:
            best_score_calnski = calinski_score
            k_calinski = k

        # Silhouette Coefficient值越大，代表聚类效果越好，取值范围为-1~1之间
        silhouette_score = metrics.silhouette_score(X, y_pred)
        # print("Silhouette Coefficient Score with gamma=", gamma, "n_clusters=", k, "score:", silhouette_score)
        if silhouette_score > best_score_silhouette:
            best_score_silhouette = silhouette_score
            k_silhouette = k

        # DBI的值越小，代表聚类效果越好，最小是0
        dbi_score = metrics.davies_bouldin_score(X, y_pred)
        # print("DBI Score with gamma=", gamma, "n_clusters=", k, "score:", dbi_score)
        if dbi_score < best_score_dbi:
            best_score_dbi = dbi_score
            k_dbi = k

    return k_calinski, k_silhouette, k_dbi


# 聚合聚类结果和原始文本摘要,保存序号和摘要文本{'topic_1': [{1:'world'}, {5, 'hello'}], 'topic_2': [{4, 'hehe'}, {18: 'haha'}]}
def combinePredictionAndAbstract(y_pred, abstract):
    result = {}
    length = len(y_pred)
    for i in range(length):
        topic = 'topic_' + str(y_pred[i])
        index_abstract = {i: abstract[i]}
        if topic not in result.keys():
            result[topic] = [index_abstract]
        else:
            result[topic].append(index_abstract)

    return result


# save to file
def saveCombinationToFile(topic_to_id, path):
    js = json.dumps(topic_to_id)
    # print("js = ", js)
    file = open(path, 'w')
    file.write(js)
    file.close()


# 保存bert向量
def saveBertEmbeddings(embeddings, path):
    print("正在保存bert词向量。。。")
    embeddings = np.array(embeddings)

    with open(path, 'w') as file_object:
        np.savetxt(file_object, embeddings, fmt='%f', delimiter=',')


# 词向量降维
def dimensionReduction(embeddings):
    print(">>>正在对词向量降维...")
    pca = PCA(n_components=3)

    embeddings = pca.fit_transform(np.array(embeddings))

    return embeddings.tolist()


# 词向量降维t-SNE
def dimensionReductionByTSNE(embeddings):
    tsne = TSNE(n_components=3, learning_rate=100, perplexity=20)
    embeddings = tsne.fit_transform(np.array(embeddings))

    return embeddings.tolist()


if __name__ == "__main__":
    print("》》》start at absa_clustering.py。。。")

    dimensions = [4, 5, 6, 7, 8]
    # cluster_number = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

    scores = []
    numbers = [9]

    # for cluster_number in range(44, 2, -1):
    for cluster_number in numbers:
        # numbers.append(cluster_number)
        # 读取不同餐厅的短文本摘要
        for restaurant_name in restaurant_names:
            print(">>>正在处理", restaurant_name)
            filepath = 'result/abstract/' + restaurant_name + '.csv'
            # 需要保证abstract没有空的
            abstract = readAbstract(filepath)
            print("abstract[20] = ", abstract[20])
            print("abstract's length = ", len(abstract))

            # 生成Bert模型
            # 提取Bert词向量，表示文本摘要
            embeddings = getBertEmbeddings(abstract)
            # print("embeddings = ", embeddings)
            print("embeddings' length = ", len(embeddings))

            # 对词向量降维
            embeddings = dimensionReductionByTSNE(embeddings)
            # embeddings = dimensionReduction(embeddings)

            # 构建聚类模型spectralClusteringPredict
            # y_pred = spectralClusteringPredict(embeddings, restaurant_name, dim)
            y_pred, calinski_score, silhouette_score, dbi_score = KMeansClusteringPredict(embeddings, restaurant_name, cluster_number)
            scores.append([calinski_score, silhouette_score, dbi_score])
            print("y_pred's length = ", len(y_pred))

            '''
            # 将聚类结果与原始文本摘要ID聚合
            result = combinePredictionAndAbstract(y_pred, abstract)
            # print("result = ", result)
            print("topics:", result.keys())

            # 将结果保存至文件
            path = 'result/clustering/3D/pred/' + 'KMeans_' + str(cluster_number) + '_' + restaurant_name + '.txt'
            saveCombinationToFile(result, path)

            # 将X保存至文件
            save_path = 'result/clustering/3D/embeddings/' + 'KMeans_' + str(cluster_number) + '_' + restaurant_name + '.txt'
            saveBertEmbeddings(embeddings, save_path)
            '''

    print("scores = ", scores)
    # print("numbers = ", numbers)

    print("》》》end of absa_clustering.py。。。")


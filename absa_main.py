#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    ***********************↓↓注意↓↓*************************
    ①全局变量（超参数）用大写字母＋下划线命名！
    ②所有函数名称使用驼峰命名法！
    ③所有局部变量使用小写＋下划线命名！
    ④所有数据、配置、路径等信息统一从absa_config.py查询
    ⑤所有测试的print代码在相应函数中打印，尽量不要在入口脚本中打印
    整体上跟之前句子级情感分析代码流程一样，只是词向量使用bert的
    ***********************↑↑注意↑↑*************************

    ***************************************************************↓↓↓↓↓↓↓↓方面级情感分析流程↓↓↓↓↓↓↓↓***************************************************************
    1.读取原始训练数据集origin_data
    1.1 统计一下输入语料的长度

    2.加载bert模型bert_model
    3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings

    4.对sentence_embeddings进行聚类，得到三个聚类中心cluster_centers，并输出到文件
    5.计算每条评论的特征向量（字符级向量）到聚类中心的距离distance_from_feature_to_cluster
    6.使用cosin距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）

    7.将review_sentiment_membership_degree拼接在word_embedding后面生成最终的词向量final_word_embeddings
    ***************************************************************↑↑↑↑↑↑↑↑方面级情感分析流程↑↑↑↑↑↑↑↑***************************************************************

    ***************************************************************↓↓↓暂时没用，后续可考虑词典增强的情感分析↓↓↓***************************************************************
    2.对origin_data进行去标点符号、去停用词、分词等得到input_texts
    5.使用input_texts和character_embedding得到每条评论的词向量word_embedding
    6.从情感词典DLUT.csv中获取情感词集SENTIMENT_WORDS_DLUT（形式为“词语word：情感值score”，如“好看：5”）
    7.构建近义词查找函数findSimilarWord(target_word)，返回近义词
    8.构建隶属函数（使用高斯函数）membershipDegree(score),返回三个模糊集（正向、负向、中性）的隶属度list
    9.读取近义词表，构建两个对称词典SYNONYM_LOOKUP_1和SYNONYM_LOOKUP_2
    10.读取boson.xlsx生成boson情感词典SENTIMENT_WORDS_BOSON
    11.使用input_texts和SENTIMENT_WORDS_DLUT、SENTIMENT_WORDS_BOSON（次优先级）、SYNONYM_LOOKUP_1和SYNONYM_LOOKUP_2（辅助）构建所有评论的情感值词典review_sentiment_score（[[-100, -5, -4, 3, 2, 5, 1],[-100, -5, -4, 3, 2, 5, 1],...[-100, -5, -4, 3, 2, 5, 1]]）(可能会比较稀疏)-100表示非情感词
    12.使用membershipDegree(score)和review_sentiment_score计算所有评论中的词语的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）
    13.将review_sentiment_membership_degree拼接在word_embedding后面生成最终的词向量final_word_embeddings
    ***************************************************************↑↑↑暂时没用，后续可考虑词典增强的情感分析↑↑↑***************************************************************
"""

import time

import sentimentAnalysis.absa_dataProcess as dp
import sentimentAnalysis.config as config
import sentimentAnalysis.absa_models as absa_models

# 几个超参数，全部大写
CONCATENATE_ADD = True  # 拼接两个字符级向量的方式-ADD
CONCATENATE_MAX = True  # 拼接两个字符级向量的方式-MAX
CONCATENATE_MEAN = True  # 拼接两个字符级向量的方式-MEAN

# 是否加入模糊情感特征
FUZZY_SENTIMENT_FEATURE = False

# 如果DEBUG为True，则只测试少部分数据
DEBUG = True


if __name__ == "__main__":
    print(">>>Begin in absa_main.py ...")
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))

    # 1.读取原始训练数据集origin_data
    origin_data, y_cols = dp.initData(DEBUG)

    # 把细粒度属性标签转为粗粒度属性标签
    # dp.processDataToTarget(origin_data)

    # 统计输入语料中不同属性-不同标签的样本数
    sample_statistics = dp.calculateSampleNumber(origin_data)
    print("sample_statistics = ", sample_statistics)

    # 2.加载bert模型bert_model
    bert_model, tokenizer = dp.createBertEmbeddingModel()

    # 3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings
    character_embeddings, sentence_embeddings = dp.getBertEmbeddings(bert_model, tokenizer, origin_data, DEBUG)

    # 4.对sentence_embeddings进行聚类，得到三个聚类中心cluster_centers，并输出到文件
    # cluster_centers = dp.getClusterCenters(sentence_embeddings)
    # 4.直接从文件中读取聚类中心向量
    cluster_centers = dp.getClusterCenterFromFile()

    # 5.计算每条评论的特征向量（字符级向量）到不同聚类中心的隶属值 distance_from_feature_to_cluster
    # 6.使用cosin余弦距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）
    membership_degrees = dp.calculateMembershipDegree(cluster_centers, character_embeddings)

    # 7.将review_sentiment_membership_degree拼接在word_embedding后面生成最终的词向量final_word_embeddings

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print(">>>End of absa_main.py...")


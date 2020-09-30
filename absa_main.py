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
    1.读取原始训练数据集origin_data,去掉换行符、空格
    1.1 统计一下输入语料的长度
    1.2 对评论语句进行补齐[pad]

    2.加载bert模型bert_model
    3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings

    4.对sentence_embeddings进行聚类，得到三个聚类中心cluster_centers，并输出到文件
    5.计算每条评论的特征向量（字符级向量）到聚类中心的距离distance_from_feature_to_cluster
    6.使用cosin距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）

    7.将review_sentiment_membership_degree拼接在wcharacter_embeddings后面生成最终的词向量final_word_embeddings

    8.构建CNN模型

    9.训练模型
    9.1 数据集按照两种方式来训练：①直接划分比例，训练集、验证集、测试集按照 7:2:1划分；②交叉验证XXXX之后再写
    ***************************************************************↑↑↑↑↑↑↑↑方面级情感分析流程↑↑↑↑↑↑↑↑***************************************************************

    ***************************************************************↓↓↓↓↓↓↓↓几个需要后期测试的点↓↓↓↓↓↓↓***************************************************************
    1.maxlen的长度是否可以增加？
    2.对原始数据进行清洗（去换行符、空格）的影响
    3.样本不均衡的处理
    4.使用交叉验证和直接划分比例看效果
    5.给执行时间较长的函数加上时间，如【3】
    6.写的差不多了可以先放到服务器上面跑着，同时改进代码
    ?.还没想到
    ***************************************************************↑↑↑↑↑↑↑↑几个需要后期测试的点↑↑↑↑↑↑↑***************************************************************

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

import absa_dataProcess as dp
import absa_config as config
import absa_models as absa_models

# 如果DEBUG为True，则只测试少部分数据
DEBUG = True

# 句子的最大长度
MAXLEN = 512

# 是否对原始数据进行清洗，如去掉换行符、空格,后续测试
CLEAN = False


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in absa_main.py ...")

    # 1.读取原始训练数据集origin_data
    print("》》》【1】读取原始训练数据集,去掉换行符、空格（测试）*******************************************************************************************************************************************************")
    origin_data, y_cols = dp.initData(DEBUG, CLEAN)

    # 把细粒度属性标签转为粗粒度属性标签
    # dp.processDataToTarget(origin_data)

    # 1.1 统计输入语料中不同属性-不同标签的样本数
    sample_statistics = dp.calculateSampleNumber(origin_data)
    print(">>>sample_statistics = ", sample_statistics)

    # 1.2 对评论文本进行[pad]操作，补到长度为512
    # origin_data = dp.textsPadding(origin_data, 512)

    # 2.加载bert模型bert_model
    print("》》》【2】加载bert模型**************************************************************************************************************************************************************************************")
    bert_model, tokenizer, token_dict = dp.createBertEmbeddingModel()

    # 3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings
    print("》》》【3】正在从bert模型获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings(此处比较耗时间注意哦~）****************************************************************************")
    character_embeddings, sentence_embeddings = dp.getBertEmbeddings(bert_model, tokenizer, origin_data, MAXLEN, DEBUG)

    # 4.对sentence_embeddings进行聚类，得到三个聚类中心cluster_centers，并输出到文件
    print("》》》【4】获取三个聚类中心**********************************************************************************************************************************************************************************")
    # cluster_centers = dp.getClusterCenters(sentence_embeddings)
    # 4.直接从文件中读取聚类中心向量
    cluster_centers = dp.getClusterCenterFromFile()

    # 5.计算每条评论的特征向量（字符级向量）到不同聚类中心的隶属值 distance_from_feature_to_cluster
    print("》》》【5、6】计算评论对聚类中心的隶属值*********************************************************************************************************************************************************************")
    # 6.使用cosin余弦距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）
    review_sentiment_membership_degree = dp.calculateMembershipDegree(cluster_centers, character_embeddings)

    # 7.将review_sentiment_membership_degree拼接在character_embeddings后面生成最终的词向量final_word_embeddings
    print("》》》【7】将隶属值拼接在原词向量上生成最终的词向量**********************************************************************************************************************************************************")
    final_word_embeddings = dp.concatenateVector(character_embeddings, review_sentiment_membership_degree)

    # 8.构建CNN模型
    print("》》》【8】构建深度学习模型**********************************************************************************************************************************************************************************")
    # bert词向量的维度时768，增加不同类别的隶属度三个维度，一共771维
    model = absa_models.createTextCNNModel(512, 771, DEBUG)

    # 9.训练模型
    print("》》》【9】训练模型******************************************************************************************************************************************************************************************")
    absa_models.trainModel(model, final_word_embeddings, origin_data, y_cols, ratio_style=True, debug=DEBUG)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of absa_main.py...")


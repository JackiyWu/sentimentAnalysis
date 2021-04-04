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

    *************************************************************↓↓↓↓↓↓↓↓方面级情感分析流程v2.0↓↓↓↓↓↓↓↓*************************************************************
    1.读取原始训练数据集origin_data,去掉换行符、空格
    1.1 统计一下输入语料的长度
    1.2 对评论语句进行补齐[pad]

    2.加载bert模型bert_model
    3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings
    3.1 保存字符向量character_embeddings、句子级向量sentence_embeddings至文件 character_embeddings.csv和sentence_embeddings.csv,格式
    3.2 直接读取两个向量文件

    4.从bert_model获取情感词典的词向量表示，并得到聚类中心，输出到文件
    4.0 从DLUT.csv中读取情感词
    4.1 获取词向量表示，并保存到文件
    4.2 读取情感词典词向量，计算得到聚类中心，聚类中心的获取使用两种方式，并验证哪种效果更好
    4.2.1 三种情感极性词向量分别聚类，各自得到一个中心向量
    4.2.2 三种情感极性词向量放在一起聚类，得到三个聚类中心

    5.计算每条评论的特征向量（字符级向量）到聚类中心的距离distance_from_feature_to_cluster
    6.使用cosin距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）

    7.将review_sentiment_membership_degree拼接在wcharacter_embeddings后面生成最终的词向量final_word_embeddings

    8.构建CNN模型

    9.训练模型
    9.1 数据集按照两种方式来训练：①直接划分比例，训练集、验证集、测试集按照 7:2:1划分；②交叉验证XXXX之后再写
    9.2 测试集的预测这里可能是有问题的，现在是每训练完一个属性就预测并打印，而不是整个模型训练完了之后才打印
    *************************************************************↑↑↑↑↑↑↑↑方面级情感分析流程v2.0↑↑↑↑↑↑↑↑*************************************************************

    ***************************************************************↓↓↓↓↓↓↓↓方面级情感分析流程↓↓↓↓↓↓↓↓***************************************************************
    1.读取原始训练数据集origin_data,去掉换行符、空格
    1.1 统计一下输入语料的长度
    1.2 对评论语句进行补齐[pad]

    2.加载bert模型bert_model
    3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings
    3.1 保存字符向量character_embeddings、句子级向量sentence_embeddings至文件 character_embeddings.csv和sentence_embeddings.csv,格式
    3.2 直接读取两个向量文件

    4.对sentence_embeddings进行聚类，得到三个聚类中心cluster_centers，并输出到文件
    5.计算每条评论的特征向量（字符级向量）到聚类中心的距离distance_from_feature_to_cluster
    6.使用cosin距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）

    7.将review_sentiment_membership_degree拼接在wcharacter_embeddings后面生成最终的词向量final_word_embeddings

    8.构建CNN模型

    9.训练模型
    9.1 数据集按照两种方式来训练：①直接划分比例，训练集、验证集、测试集按照 7:2:1划分；②交叉验证XXXX之后再写
    9.2 测试集的预测这里可能是有问题的，现在是每训练完一个属性就预测并打印，而不是整个模型训练完了之后才打印
    ***************************************************************↑↑↑↑↑↑↑↑方面级情感分析流程↑↑↑↑↑↑↑↑***************************************************************

    ***************************************************************↓↓↓↓↓↓↓↓几个需要后期测试的点↓↓↓↓↓↓↓***************************************************************
    1.maxlen的长度是否可以增加？
    2.对原始数据进行清洗（去换行符、空格）的影响
    3.样本不均衡的处理
    4.使用交叉验证和直接划分比例看效果
    5.给执行时间较长的函数加上时间，如【3】
    6.写的差不多了可以先放到服务器上面跑着，同时改进代码
    7.去掉bert的[CLS] [SEP]符号
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

# @title Main function

import time

import absa_dataProcess as dp
import absa_config as config
import absa_models as absa_models

# 如果DEBUG为True，则只测试少部分数据
DEBUG = False
DEBUG_ONLINE = False

# 句子的最大长度
MAXLEN = 512

# 是否对原始数据进行清洗，如去掉换行符、空格,后续测试
CLEAN_ENTER = True
CLEAN_SPACE = False

# 向量维度
EMBEDDING_DIM_CHARACTER = 768
EMBEDDING_DIM_MEMBERSHIP = 3
EMBEDDING_DIM_FINAL = 771


if __name__ == "__main__":
    start_time = time.time()
    print("Start time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    print(" Begin in absa_main.py ...")

    # 1.读取原始训练数据集origin_data
    print("》》》【1】读取原始训练数据集,去掉换行符、空格（测试）*******************************************************************************************************************************************************")
    # origin_data, y_cols, y = dp.initData(DEBUG, CLEAN_ENTER, CLEAN_SPACE)
    # 1.1 读取原始训练数据集的labels
    # y_cols_name, y_train, y_validation = dp.initDataLabels(DEBUG)

    # 把细粒度属性标签转为粗粒度属性标签
    # dp.processDataToTarget(origin_data)

    # 1.1 统计输入语料中不同属性-不同标签的样本数
    # sample_statistics = dp.calculateSampleNumber(origin_data)
    # print(">>>sample_statistics = ", sample_statistics)

    # 1.2 对评论文本进行[pad]操作，补到长度为512
    # origin_data = dp.textsPadding(origin_data, 512)

    # 2.加载bert模型bert_model
    print("》》》【2】加载bert模型**************************************************************************************************************************************************************************************")
    # bert_model, tokenizer, token_dict = dp.createBertEmbeddingModel()
    '''
    if DEBUG:
        cluster_centers_path = config.cluster_center_validation_1
        character_embeddings_path = config.character_embeddings_validation
        sentence_embeddings_path = config.sentence_embeddings_validation
        membership_degree_path = config.membership_degree_validation
        X_train_path = config.final_word_embeddings_validation
        final_word_embeddings_path = config.final_word_embeddings_validation
    else:
        cluster_centers_path = config.cluster_center_train_1
        character_embeddings_path = config.character_embeddings_train
        membership_degree_path = config.membership_degree_train
        final_word_embeddings_path = config.final_word_embeddings_train
        X_train_path = config.final_word_embeddings_train
    '''

    # 3.从bert_model获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings
    print("》》》【3】正在从bert模型获取origin_data对应的字符向量character_embeddings、句子级向量sentence_embeddings(此处比较耗时间注意哦~）****************************************************************************")
    # character_embeddings, sentence_embeddings = dp.getBertEmbeddings(bert_model, tokenizer, origin_data, MAXLEN, DEBUG)
    # 3.1 从文件中读取character_embeddings, sentence_embeddings
    # character_embeddings = dp.getCharacterEmbeddings(character_embeddings_path)
    # sentence_embeddings = dp.getSentenceEmbeddings(sentence_embeddings_path)

    # 4.对sentence_embeddings进行聚类，得到三个聚类中心cluster_centers，并输出到文件
    print("》》》【4】获取三个聚类中心**********************************************************************************************************************************************************************************")
    # cluster_centers = dp.getClusterCenters(sentence_embeddings, cluster_centers_path)
    # print("cluster_centers' length = ", len(cluster_centers))

    # 4.从bert_model获取情感词向量sentiment_word_embeddings
    '''
    sentiment_words_path = config.sentiment_dictionary_dut
    bert_path = config.bert_path
    cluster_centers_path = config.cluster_center_train_1
    cluster_centers = dp.getClusterCentersV2(sentiment_words_path, cluster_centers_path, bert_path, DEBUG)
    '''
    # 4.1 直接从文件中读取聚类中心向量
    # cluster_centers = dp.getClusterCenterFromFile(cluster_centers_path)
    # print("cluster_centers' length = ", len(cluster_centers))
    # print("cluster_centers[0]' length = ", len(cluster_centers[0]))
    # print("cluster_centers[0][0]' length = ", len(cluster_centers[0][0]))

    # 5.计算每条评论的特征向量（字符级向量）到不同聚类中心的隶属值 distance_from_feature_to_cluster
    print("》》》【5、6】计算评论对聚类中心的隶属值*********************************************************************************************************************************************************************")
    # 6.使用cosin余弦距离来定义隶属函数,根据distance_from_feature_to_cluster和隶属函数计算特征向量对三个类别的隶属值review_sentiment_membership_degree([])（三维隶属值，表示负向、中性、正向）
    # review_sentiment_membership_degree = dp.calculateMembershipDegree(cluster_centers, character_embeddings)
    # 计算并保存评论文本的隶属度
    # review_sentiment_membership_degree = dp.calculateAndSaveMembershipDegree(cluster_centers, character_embeddings_path, membership_degree_path, DEBUG)

    # 7.将review_sentiment_membership_degree拼接在character_embeddings后面生成最终的词向量final_word_embeddings
    print("》》》【7】将隶属值拼接在原词向量上生成最终的词向量**********************************************************************************************************************************************************")
    # final_word_embeddings_path = config.final_word_embeddings_validation
    # final_word_embeddings = dp.concatenateVector(character_embeddings, review_sentiment_membership_degree, final_word_embeddings_path)
    # 将final_word_embeddings存入文件
    # dp.saveFinalEmbeddings(final_word_embeddings, final_word_embeddings_path)
    # 训练集的数据太大，只能一边读取 一边拼接 一边存入文件
    # dp.saveFinalEmbeddingLittleByLittle(review_sentiment_membership_degree, character_embeddings_path, final_word_embeddings_path, DEBUG)
    # final_word_embeddings = dp.getFinalEmbeddings(final_word_embeddings_path, DEBUG)

    # 8.加载bert模型
    print("》》》【8】构建bert学习模型**********************************************************************************************************************************************************************************")
    # bert词向量的维度是768，增加不同类别的隶属度三个维度，一共771维
    # bert_model = absa_models.createBertEmbeddingModel()

    X, y_cols, Y = dp.initDataForBert(config.meituan_train_new, DEBUG, CLEAN_ENTER, CLEAN_SPACE)
    X_validation, y_cols_validation, Y_validation = dp.initDataForBert(config.meituan_validation_new, DEBUG, CLEAN_ENTER, CLEAN_SPACE)

    # 加载tokenizer
    tokenizer = absa_models.get_tokenizer()

    experiment_name = ""

    # 9.训练模型
    print("》》》【9】训练模型******************************************************************************************************************************************************************************************")
    # X_train_path = config.character_embeddings_train_tuned
    # X_validation_path = config.character_embeddings_validation_tuned
    # if DEBUG_ONLINE:
    #     X_train_path = X_validation_path
    #     y_train = y_validation
    # epochs = [1]
    epochs = [3]
    # batch_sizes = [10]
    batch_sizes = [96]
    times = 1  # 设置为1是为了测试看结果
    print("training times = ", times)
    model_name = "BertCNNBiGRUModel_multiGPU_20210326"
    # batch_size_validation = 30
    batch_size_validation = 256

    if model_name.startswith("BertCNNModel"):
        filters = [128]
        # window_sizes = [6]
        window_sizes = [4, 5, 6, 7]
        for cnn_filter in filters:
            for window_size in window_sizes:
                for batch_size in batch_sizes:
                    for epoch in epochs:
                        experiment_name = model_name + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size) + "_filter_" + str(filters) + "_windowSize_" + str(window_size)
                        print("experiment_name = ", experiment_name)
                        for i in range(times):
                            model = absa_models.createBertCNNModel(cnn_filter, window_size)
                            absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    if model_name.startswith("BertMultiCNNModel"):
        filters1 = [128]
        window_sizes1 = [6]
        filters2 = [128]
        window_sizes2 = [5]
        # window_sizes = [4, 5, 6, 7]
        for cnn_filter1 in filters1:
            for window_size1 in window_sizes1:
                for cnn_filter2 in filters2:
                    for window_size2 in window_sizes2:
                        for batch_size in batch_sizes:
                            for epoch in epochs:
                                experiment_name = model_name + "_filter1_" + str(cnn_filter1) + "_windowSize1_" + str(window_size1) + "_filter2_" + str(cnn_filter2) + "_windowSize2_" + str(window_size2) + "_batchSize_" + str(batch_size) + "_epoch_" + str(epoch)
                                print("experiment_name = ", experiment_name)
                                for i in range(times):
                                    model = absa_models.createBertMultiCNNModel(cnn_filter1, window_size1, cnn_filter2, window_size2)
                                    absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertSeparableCNNModel"):
        for batch_size in batch_sizes:
            for epoch in epochs:
                experiment_name = model_name + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                print("experiment_name = ", experiment_name)
                for i in range(times):
                    model = absa_models.createBertSeparableCNNModel()
                    absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertGRUModel"):
        gru_output_dim_1 = [256]
        for dim_1 in gru_output_dim_1:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    experiment_name = model_name + "_gru_dim_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                    print("experiment_name = ", experiment_name)
                    for i in range(times):
                        model = absa_models.createBertGRUModel(dim_1)
                        absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertOriginGRUModel"):
        gru_output_dim_1 = [256]
        for dim_1 in gru_output_dim_1:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    experiment_name = model_name + "_gru_dim_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                    print("experiment_name = ", experiment_name)
                    for i in range(times):
                        model = absa_models.createBertOriginGRUModel(dim_1)
                        absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertSeparableCNNBiGRUModel"):
        filters = [200]
        window_sizes_1 = [4]
        window_sizes_2 = [5]
        window_sizes_3 = [6]
        gru_output_dim_1 = [128]
        gru_output_dim_2 = [64]
        for cnn_filter in filters:
            for window_size_1 in window_sizes_1:
                for window_size_2 in window_sizes_2:
                    for window_size_3 in window_sizes_3:
                        for dim_1 in gru_output_dim_1:
                            for dim_2 in gru_output_dim_2:
                                for batch_size in batch_sizes:
                                    for epoch in epochs:
                                        experiment_name = model_name + "_filter_" + str(cnn_filter) + "_window_size1_" + str(window_size_1) + "_window_size2_" + str(window_size_2) + "_window_size3_" + str(window_size_3) + "_gru_dim1_" + str(dim_1) + "_gru_dim2_" + str(dim_2) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                                        print("experiment_name = ", experiment_name)
                                        for i in range(times):
                                            model = absa_models.createBertSeparableCNNBiGRUModel(cnn_filter, window_size_1, window_size_2, window_size_3)
                                            absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertMultiCNNBiGRUModel"):
        filters = [200]
        window_sizes_1 = [3, 4, 5]
        window_sizes_2 = [4, 5, 6]
        gru_output_dim_1 = [64, 128, 256]
        gru_output_dim_2 = [64]
        for cnn_filter in filters:
            for window_size_1 in window_sizes_1:
                for window_size_2 in window_sizes_2:
                    for dim_1 in gru_output_dim_1:
                        for dim_2 in gru_output_dim_2:
                            for batch_size in batch_sizes:
                                for epoch in epochs:
                                    experiment_name = model_name + "_filter_" + str(cnn_filter) + "_window_size1_" + str(window_size_1) + "_window_size2_" + str(window_size_2) + "_gru_dim1_" + str(dim_1) + "_gru_dim2_" + str(dim_2) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                                    print("experiment_name = ", experiment_name)
                                    for i in range(times):
                                        model = absa_models.createBertMultiCNNBiGRUModel(cnn_filter, window_size_1, window_size_2, dim_1, dim_2)
                                        absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertLSTMModel"):
        dims_1 = [64]
        for dim_1 in dims_1:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    experiment_name = model_name + "_dim1_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                    print("experiment_name = ", experiment_name)
                    for i in range(times):
                        model = absa_models.createBertLSTMModel(dim_1)
                        absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertBiLSTMModel"):
        dims_1 = [128]
        dims_2 = [64]
        for dim_1 in dims_1:
            for dim_2 in dims_2:
                for batch_size in batch_sizes:
                    for epoch in epochs:
                        experiment_name = model_name + "_dim1_" + str(dim_1) + "_dim2_" + str(dim_2) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                        print("experiment_name = ", experiment_name)
                        for i in range(times):
                            model = absa_models.createBertBiLSTMModel(dim_1, dim_2)
                            absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertMLPModel"):
        dense_dims = [64]
        for dim in dense_dims:
            for batch_size in batch_sizes:
                for epoch in epochs:
                    experiment_name = model_name + "_dim_" + str(dim) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                    print("experiment_name = ", experiment_name)
                    for i in range(times):
                        model = absa_models.createBertMLPModel()
                        absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("BertCNNBiGRUModel"):
        filters = [256]
        window_sizes = [4]
        gru_output_dim_1 = [256]
        for cnn_filter in filters:
            for window_size in window_sizes:
                for dim_1 in gru_output_dim_1:
                    for batch_size in batch_sizes:
                        for epoch in epochs:
                            experiment_name = model_name + "_filter_" + str(cnn_filter) + "_window_size_" + str(window_size) + "_gru_dim_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                            print("experiment_name = ", experiment_name)
                            for i in range(times):
                                print("current times is ", i)
                                model = absa_models.createBertCNNBiGRUModel(cnn_filter, window_size, dim_1)
                                absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)
    elif model_name.startswith("FuzzyBertCNNBiGRUModel"):
        # 直接读取评论文本的隶属度
        membership_degree_path_train = config.membership_degree_train
        review_sentiment_membership_degree_train = dp.getMembershipDegrees(membership_degree_path_train)
        membership_degree_path_validation = config.membership_degree_validation
        review_sentiment_membership_degree_validation = dp.getMembershipDegrees(membership_degree_path_validation)
        filters = [512, 1024]
        window_sizes = [4]
        gru_output_dim_1 = [128, 256, 512]
        for cnn_filter in filters:
            for window_size in window_sizes:
                for dim_1 in gru_output_dim_1:
                    for batch_size in batch_sizes:
                        for epoch in epochs:
                            experiment_name = model_name + "_filter_" + str(cnn_filter) + "_window_size_" + str(window_size) + "_gru_dim1_" + str(dim_1) + "_epoch_" + str(epoch) + "_batchSize_" + str(batch_size)
                            print("experiment_name = ", experiment_name)
                            for i in range(times):
                                model = absa_models.createFuzzyBertCNNBiGRUModel(cnn_filter, window_size, dim_1)
                                absa_models.trainBert(experiment_name, model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation,
                                                      membership_train=review_sentiment_membership_degree_train, membership_validation=review_sentiment_membership_degree_validation, debug=DEBUG)
    else:
        pass

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of absa_main.py...")


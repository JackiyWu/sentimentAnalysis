#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import absa_dataProcess as dp
import absa_config as config
import absa_models as absa_models

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 如果DEBUG为True，则只测试少部分数据
DEBUG = True
DEBUG_ONLINE = True

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
    X, y_cols, Y = dp.initDataForBert(config.meituan_train_new, DEBUG, CLEAN_ENTER, CLEAN_SPACE)
    X_validation, y_cols_validation, Y_validation = dp.initDataForBert(config.meituan_validation_new, DEBUG, CLEAN_ENTER, CLEAN_SPACE)

    # 加载bert模型
    bert_model = absa_models.createBert()

    # 加载tokenizer
    tokenizer = absa_models.get_tokenizer()

    experiment_name = ""
    model_name = "bert"

    batch_size = 20
    batch_size_validation = 128
    # 训练模型
    absa_models.trainBert(experiment_name, bert_model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, 3, 20, batch_size_validation)


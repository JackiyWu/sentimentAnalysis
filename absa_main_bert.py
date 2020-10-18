#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import absa_dataProcess as dp
import absa_config as config
import absa_models as absa_models

from keras.models import load_model
from keras.layers import Lambda
from keras_bert import get_custom_objects


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 如果DEBUG为True，则只测试少部分数据
DEBUG = False
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
    # bert_model = absa_models.createBertCNN()

    # 加载tokenizer
    tokenizer = absa_models.get_tokenizer()

    experiment_name = ""
    model_name = "bert"

    batch_size = 20
    batch_size_validation = 128
    epoch = 5
    # 训练模型
    absa_models.trainBert(experiment_name, bert_model, X, Y, y_cols, X_validation, Y_validation, model_name, tokenizer, epoch, batch_size, batch_size_validation, DEBUG)

    # 保存bert模型
    print(">>>正在保存bert模型。。。")
    bert_model.save(config.tuned_bert_model)
    print(">>>bert模型已保存。。。")

    print(">>>加载bert模型。。。")
    bert_model = load_model(config.tuned_bert_model, custom_objects=get_custom_objects())
    bert_model.summary()
    print(">>>bert模型加载结束。。。")

    # 提取词向量
    path = "train"
    absa_models.getAndSaveBertEmbeddingsAfterTuned(bert_model, X, path, tokenizer)

    path = "validation"
    absa_models.getAndSaveBertEmbeddingsAfterTuned(bert_model, X_validation, path, tokenizer)

    end_time = time.time()
    print("End time : ",  time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))
    total_time = end_time - start_time
    print("Consume total time:", total_time, "s")

    print("》》》End of absa_main_bert.py...")


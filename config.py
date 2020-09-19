#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

data_path = os.path.abspath('C:/cygwin/home/wujie/src/ML_yt/panyang') + "/data"
model_path = data_path + "/model/"
raw_data_path = data_path + "/0526/0526rawL3.csv"
train_data_path = data_path + "/train/0526trainL3.csv"
test_data_path = data_path + "/test/0526testL3.csv"
predict_data_path = data_path + "/predict/0611predictL3_benz.csv"
result_data_path = data_path + "/result/0611resultL3_benz.csv"


sentiment_dictionary = "C:\desktop\Research\情感词典\DLUT-Emotionontology-master\情感词汇\情感词汇本体\情感词汇本体.xlsx"

pre_word_embedding = "C:\desktop\Coding\preWordEmbedding\sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"

# 百度点石数据集
baidu_sentiment = "C:\desktop\Research\DataSet\百度点石大赛\data_train.csv"
baidu_sentiment_test = "C:\desktop\Research\DataSet\百度点石大赛\data_test.csv"


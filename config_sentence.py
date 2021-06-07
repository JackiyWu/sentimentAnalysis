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


sentiment_dictionary = "sentiVocab/DLUT.xls"
sentiment_dictionary_csv = "sentiVocab/DLUT.csv"

pre_word_embedding = "config/preEmbeddings/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"
pre_word_embedding_2 = "config/preEmbeddings/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5"
pre_word_embedding_3 = "config/preEmbeddings/sgns.wiki.word"

# 百度点石数据集
baidu_sentiment = "datasets/baidu/data_train.csv"
baidu_sentiment_test = "datasets/baidu/data_test.csv"

synonym_txt = "sentiVocab/dict_synonym.txt"
synonym_xlsx = "sentiVocab/dict_synonym.xlsx"

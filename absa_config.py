#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 情感词典
sentiment_dictionary_dut = "sentiVocab/DLUT.csv"
sentiment_dictionary_boson = "sentiVocab/BosonNLP_sentiment_score.txt"

# 数据集
meituan_train = "datasets/meituan/sentiment_analysis_training_set.csv"
meituan_test = "datasets/meituan/sentiment_analysis_test_set.csv"
meituan_validation = "datasets/meituan/sentiment_analysis_validation_set.csv"

# 数据集-粗粒度（5个属性）
meituan_train_new = "datasets/meituan/sentiment_analysis_training_set_new.csv"
meituan_test_new = "datasets/meituan/sentiment_analysis_test_set_new.csv"
meituan_validation_new = "datasets/meituan/sentiment_analysis_validation_set_new.csv"

# keras_bert
bert_config_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = 'config/keras_bert/chinese_L-12_H-768_A-12/vocab.txt'

# 结果保存
cluster_center = 'result/cluster_center.csv'

character_embeddings_validation = 'result/character_embeddings_validation.txt'
character_embeddings_train = 'result/character_embeddings_train.txt'
sentence_embeddings_validation = 'result/sentence_embeddings_validation.txt'
sentence_embeddings_train = 'result/sentence_embeddings_train.txt'


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import codecs

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import keras
from keras.utils.np_utils import to_categorical

from keras_bert import Tokenizer, load_trained_model_from_checkpoint

import sentimentAnalysis.absa_config as config

# 一些超参数
TOKEN_DICT = {}


# 创建bert+fc模型
def createBertFcModel():
    pass


# 创建bert模型
def createBertEmbeddingModel():
    with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            TOKEN_DICT[token] = len(TOKEN_DICT)

    model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path)

    return model


# 根据bert模型输出词向量
def getBertCharacterEmbedding(bert_model, origin_data):
    pass


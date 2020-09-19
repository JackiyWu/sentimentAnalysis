# !/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

stop_words = []
f = open(file='C:\cygwin\home\wujie\src\ML_yt\panyang\stopwords.txt', mode='r', encoding='utf-8')              #文件为123.txt
sourceInLines = f.readlines()
f.close()
for line in sourceInLines:
    temp = line.strip('\n')
    stop_words.append(temp)


def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def seg_words(contents):
    contents_segs = list()
    for content in contents:
        content = str(content)
        rcontent = content.replace("\r\n", " ").replace("\n", " ")
        segs = [word for word in jieba.cut(rcontent) if word not in stop_words]
        contents_segs.append(" ".join(segs))
    return contents_segs


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def get_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def get_precision_score(y_true, y_pred):
    return precision_score(y_true, y_pred, average='weighted')


def get_recall_score(y_true, y_pred):
    return recall_score(y_true, y_pred, average='weighted')

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

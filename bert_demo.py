#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, load_vocabulary  # 超参数

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

import absa_config as config

maxlen = 100
batch_size = 16
droup_out_rate = 0.5
learning_rate = 1e-5
epochs = 15

path_prefix = "datasets/hotel"
# 预训练模型目录
config_path = config.bert_config_path
checkpoint_path = config.bert_checkpoint_path
dict_path = config.bert_dict_path

# 读取数据
neg = pd.read_excel(path_prefix + "/neg.xlsx", header=None)
pos = pd.read_excel(path_prefix + "/pos.xlsx", header=None)

# print(neg.head())

print("neg's type = ", type(neg))
print("neg's shape = ", neg.shape)
print("neg's length = ", len(neg))
print("neg[0] = ", neg.iloc[0])
print("neg[1] = ", neg.iloc[1])

print("pos's type = ", type(pos))
print("pos[0] = ", pos.iloc[0])
print("pos[5] = ", pos.iloc[5])
print("pos's length = ", len(pos))
print("pos's shape = ", pos.shape)

# 构建训练数据
data = []

for d in neg[0]:
    data.append((d, 0))

for d in pos[0]:
    data.append((d, 1))

print(">>>数据读取完毕了哦~~~")

# 读取字典
token_dict = load_vocabulary(dict_path)
# 建立分词器
tokenizer = Tokenizer(token_dict)

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=batch_size):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


# trainable设置True对Bert进行微调
# 默认不对Bert模型进行调参
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, trainable=True)

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)
x = Dropout(droup_out_rate)(x)
p = Dense(1, activation='sigmoid')(x)

model = Model([x1_in, x2_in], p)
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate),
    metrics=['accuracy']
)
model.summary()

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

print("train_D's length = ", len(train_D))
print("valid_D's length = ", len(valid_D))

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=epochs,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)

print(">>>end of bert_demo.py...")


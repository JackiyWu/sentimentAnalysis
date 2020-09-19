from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam

# from bert4keras.bert import load_pretrained_model
# from bert4keras.utils import SimpleTokenizer, load_vocab
from bert4keras.models import build_transformer_model

import numpy as np

import pandas as pd

# 序列最大长度
maxlen = 100
config_path = 'C:\desktop\Coding\preWordEmbedding\chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'C:\desktop\Coding\preWordEmbedding\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'C:\desktop\Coding\preWordEmbedding\chinese_L-12_H-768_A-12/vocab.txt'


# neg = pd.read_excel('datasets/neg.xls', header=None)
# pos = pd.read_excel('datasets/pos.xls', header=None)

model = build_transformer_model(
    config_path,
    checkpoint_path
)

output = Lambda(lambda x: x[:, 0])(model.output)
output = Dense(1, activation='sigmoid')(output)
model = Model(model.input, output)

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(1e-5), {1000: 1e-5, 2000: 6e-5}),
    metrics=['accuracy']
)
print(model.summary())
'''
model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=10,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
'''


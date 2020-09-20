import pandas as pd
import codecs, gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
import config

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 预训练好的模型
config_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'config/keras_bert/chinese_L-12_H-768_A-12/vocab.txt'


# bert
def get_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    for l in bert_model.layers:
        l.trainable = True
    T1 = Input(shape=(None,))
    T2 = Input(shape=(None,))
    T = bert_model([T1, T2])
    T = Lambda(lambda x: x[:, 0])(T)
    output = Dense(4, activation='softmax')(T)
    model = Model([T1, T2], output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),
        metrics=['accuracy']
    )
    model.summary()
    return model


if __name__ == "__main__":
    print(">>>begin in bert_demo3 ...")

    # bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    model = get_model()

    # print(model.summary())

    print(">>>end of bert_demo3 ...")


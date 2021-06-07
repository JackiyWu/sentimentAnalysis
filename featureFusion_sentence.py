#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tensorflow.keras.layers import Flatten, Input, Dense, Dropout, concatenate, Activation, LSTM, Bidirectional, GRU, Attention, Concatenate
from tensorflow.keras.layers import BatchNormalization, Conv1D, Conv2D, MaxPool1D, MaxPool2D, Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
# from tensorflow.keras import backend as K
from keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Permute
from tensorflow import multiply

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import keras
from tensorflow.keras import optimizers
import tensorflow as tf

import numpy as np
import os
from sklearn.model_selection import train_test_split
import time
import codecs
import csv
import sys

# import dataProcess as dp
import config_sentence


'''
# 一些超参数
max_words = 50000  # 词典的长度，后期可以测试？？
maxlen = 0  # 样本的长度，后期可以测试不同长度对结果的影响，参考hotelDataEmbedding.py的处理？？
embedding_dim = 128  # 词嵌入的维度，后期可以测试？？
embeddings = []  # 词嵌入，后期可以使用预训练词向量??
dealed_train = []  # 输入语料，训练集
dealed_val = []  # 输入语料，验证集
dealed_test = []  # 输入语料，测试集
y_cols = []
'''


# 创建融合模型
def create_fusion_model_new(cnn_maxlen, dict_length, cnn_filter, embedding_matrix, window_size, full_connect):
    print("开始构建新的CNN融合模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_determinacy_0 = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        inputs_determinacy = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_determinacy_0)
        inputs_fuzzy = Input(shape=(cnn_maxlen, 3,), name='input_fuzzy')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
        inputs_concat = concatenate([inputs_determinacy, inputs_fuzzy], axis=-1)
        # inputs = Dense(full_connect, activation='relu')(inputs_concat)
        x_cnn = Conv1D(cnn_filter, window_size, padding='same', activation='relu', name='conv1')(inputs_concat)
        x_cnn = MaxPool1D(name='pool1')(x_cnn)
        x_cnn = Flatten(name='flatten')(x_cnn)

        x = Dropout(0.5, name="dropout2")(x_cnn)
        x = Dense(3, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=[inputs_determinacy_0, inputs_fuzzy], outputs=x, name='fusion_model')
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    print(fusion_model.summary())

    return fusion_model


# 创建融合模型 mlp
def create_fusion_model_mlp(maxlen, dict_length, embedding_matrix, full_connected):
    print("开始构建新的融合MLP模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_determinacy_0 = Input(shape=(maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        inputs_determinacy = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_determinacy_0)
        inputs_fuzzy = Input(shape=(maxlen, 3,), name='input_fuzzy')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
        inputs = concatenate([inputs_determinacy, inputs_fuzzy], axis=-1)
        x = Flatten(name='flatten')(inputs)

        x = Dense(full_connected, activation='relu', name='dense3')(x)
        x = Dropout(0.5, name="dropout2")(x)
        x = Dense(3, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=[inputs_determinacy_0, inputs_fuzzy], outputs=x, name='fusion_model')
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    print(fusion_model.summary())

    return fusion_model


# 创建融合模型 gru
def create_fusion_model_gru(maxlen, dict_length, embedding_matrix, dim):
    print("开始构建新的融合模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_determinacy_0 = Input(shape=(maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        inputs_determinacy = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_determinacy_0)
        inputs_fuzzy = Input(shape=(maxlen, 3,), name='input_fuzzy')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
        inputs = concatenate([inputs_determinacy, inputs_fuzzy], axis=-1)

        x_gru = Bidirectional(GRU(dim))(inputs)

        x = Dropout(0.5, name="dropout2")(x_gru)
        x = Dense(3, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=[inputs_determinacy_0, inputs_fuzzy], outputs=x, name='fusion_model')
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    print(fusion_model.summary())

    return fusion_model


# 创建融合模型
def create_fusion_model_lstm(cnn_maxlen, dict_length, embedding_matrix, dim):
    print("开始构建新的融合LSTM模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_determinacy_0 = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        inputs_determinacy = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_determinacy_0)
        inputs_fuzzy = Input(shape=(cnn_maxlen, 3,), name='input_fuzzy')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
        inputs = concatenate([inputs_determinacy, inputs_fuzzy], axis=-1)

        x_lstm = Bidirectional(LSTM(dim))(inputs)
        # x_lstm = Bidirectional(LSTM(32, return_sequences=False))(x_lstm)

        x = Dropout(0.5, name="dropout2")(x_lstm)
        x = Dense(3, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=[inputs_determinacy_0, inputs_fuzzy], outputs=x, name='fusion_lstm_model')
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    print(fusion_model.summary())

    return fusion_model


# 创建融合模型
# filters格式为列表，如[64, 32]，方便后续调优
# def create_fusion_model(fuzzy_maxlen, cnn_maxlen, dict_length, filters):
def create_fusion_model(fuzzy_maxlen, cnn_maxlen, dict_length, filter, embedding_matrix, window_size, full_connected):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # define our MLP network
        inputs_fuzzy = Input(shape=(fuzzy_maxlen,), name='input_fuzzy')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
        x_fuzzy = Dense(16, activation='linear', name='dense1_fuzzy')(inputs_fuzzy)
        # x_mlp = Dense(4, activation='relu', name='dense2_mlp')(x_mlp)

        # define our CNN
        inputs_cnn = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        # x_cnn = Embedding(input_dim=dict_length, output_dim=128, name='embedding_cnn')(inputs_cnn)
        x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_cnn)
        x_cnn = Conv1D(filter, window_size, activation='relu', name='conv1')(x_cnn)
        # x_cnn = BatchNormalization(axis=chanDim)(x_cnn)
        x_cnn = MaxPool1D(name='pool1')(x_cnn)
        x_cnn = Flatten(name='flatten')(x_cnn)

        # 融合两个输入
        x_concatenate = concatenate([x_fuzzy, x_cnn], name='fusion')

        x = Dense(full_connected, activation='relu', name='dense3')(x_concatenate)
        # x = Dropout(0.4, name="dropout1")(x)
        # x = Dense(32, activation='relu', name='dense4')(x)
        x = Dropout(0.5, name="dropout2")(x)
        x = Dense(3, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=[inputs_fuzzy, inputs_cnn], outputs=x, name='fusion_model')

        adam = optimizers.Adam(learning_rate=0.001)

        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    print(fusion_model.summary())

    # return our model
    return fusion_model


# 训练模型
'''
def train_model(model, train, val, train_x_fuzzy, train_x_cnn, test_x_fuzzy, test_x, val_x_fuzzy, val_x, y_cols, epoch,
                experiment_id, batch_size, learning_rate, balanced, model_name, debug=False, folds=1):
'''
def train_model(model, train, val, train_x_fuzzy, train_x_cnn, test_x_fuzzy, test_x, val_x_fuzzy, val_x, y_cols, epoch,
                experiment_id, batch_size, learning_rate, balanced, model_name, val_x_fuzzy_medical, val_x_medical, val_medical,
                val_x_fuzzy_financial, val_x_financial, val_financial, val_x_fuzzy_traveling, val_x_traveling, val_traveling, debug=False, folds=1):
    print(">>>in train_model function...")

    experiment_id = "fusion_model_" + experiment_id

    # print(mlp_model.summary())
    # print("检查输入数据：")
    # print("train_x_fuzzy.shape = ", train_x_fuzzy.shape)
    # print("test_x.shape = ", test_x.shape)
    # print("val_x.shape = ", val_x.shape)
    # print("y_cols = ", y_cols)

    F1_scores = 0
    F1_score = 0
    result = {}

    train_y = train["label"]
    val_y = val["label"]
    val_y_medical = val_medical["label"]
    val_y_financial = val_financial["label"]
    val_y_traveling = val_traveling["label"]
    '''
    '''
    y_val_pred = 0
    y_test_pred = 0
    #         epochs=[5,10]   , stratify=train_y
    print("train_x_fuzzy.shape:", train_x_fuzzy.shape)
    print("train_x_cnn.shape:", train_x_cnn.shape)

    print("train_y.shape:", train_y.shape)
    print("val_x.shape:", val_x.shape)
    print("val_y.shape:", val_y.shape)
    # print("train_y = ", train_y)
    # print("val_y = ", val_y)

    # 计算class_weight
    # cw = class_weight.compute_class_weight("balanced", np.unique(train_y), train_y)
    # cw = class_weights[index]

    for i in range(folds):
        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)
        # print("y_train_onehot's shape = ", y_train_onehot.shape)
        # print("y_val_onehot's shape = ", y_val_onehot.shape)
        # print("train_x_cnn = ", train_x_cnn)
        # print("train_x_cnn's shape = ", train_x_cnn.shape)
        if balanced:
            history = model.fit([train_x_fuzzy, train_x_cnn], y_train_onehot, epochs=epoch, verbose=2,
                            batch_size=batch_size, validation_data=([val_x_fuzzy, val_x], y_val_onehot))
                            # class_weight=cw)
        else:
            history = model.fit([train_x_fuzzy, train_x_cnn], y_train_onehot, epochs=epoch, verbose=2,
                                batch_size=batch_size, validation_data=([val_x_fuzzy, val_x], y_val_onehot))

        # 预测验证集和测试集
        # print("val_x = ", val_x)
        print("val_x_fuzzy.shape:", val_x_fuzzy.shape)
        print("val_x.shape:", val_x.shape)
        y_val_pred = model.predict([val_x_fuzzy, val_x])
        print("val_x_fuzzy_medical.shape:", val_x_fuzzy_medical.shape)
        print("val_x_medical.shape:", val_x_medical.shape)
        y_val_pred_medical = model.predict([val_x_fuzzy_medical, val_x_medical])
        print("val_x_fuzzy_financial.shape:", val_x_fuzzy_financial.shape)
        print("val_x_financial.shape:", val_x_financial.shape)
        y_val_pred_financial = model.predict([val_x_fuzzy_financial, val_x_financial])
        print("val_x_fuzzy_traveling.shape:", val_x_fuzzy_traveling.shape)
        print("val_x_traveling.shape:", val_x_traveling.shape)
        y_val_pred_traveling = model.predict([val_x_fuzzy_traveling, val_x_traveling])
        '''
        '''
        # y_test_pred += model.predict([test_x_fuzzy, test_x])
        # 把预测结果保存入文件
        print(">>>将预测的结果存入csv文件中。。。")
        save_predict_result_to_csv(val["review"], y_val_pred)

    y_val_pred = np.argmax(y_val_pred, axis=1)

    y_val_pred_medical = np.argmax(y_val_pred_medical, axis=1)
    y_val_pred_financial = np.argmax(y_val_pred_financial, axis=1)
    y_val_pred_traveling = np.argmax(y_val_pred_traveling, axis=1)
    '''
    '''

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("val_y = ", val_y)
    # print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(val_y, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)
    precision_medical, recall_medical, fscore_medical, support_medical = score(val_y_medical, y_val_pred_medical)
    precision_financial, recall_financial, fscore_financial, support_financial = score(val_y_financial, y_val_pred_financial)
    precision_traveling, recall_traveling, fscore_traveling, support_traveling = score(val_y_traveling, y_val_pred_traveling)
    '''
    '''

    report = classification_report(val_y, y_val_pred, digits=4, output_dict=True)

    report_medical = classification_report(val_y_medical, y_val_pred_medical, digits=4, output_dict=True)
    report_financial = classification_report(val_y_financial, y_val_pred_financial, digits=4, output_dict=True)
    report_traveling = classification_report(val_y_traveling, y_val_pred_traveling, digits=4, output_dict=True)
    '''
    '''

    print(report)

    F1_score = f1_score(y_val_pred, val_y, average='macro')
    F1_scores += F1_score

    F1_score_medical = f1_score(y_val_pred_medical, val_y_medical, average='macro')
    F1_score_financial = f1_score(y_val_pred_financial, val_y_financial, average='macro')
    F1_score_traveling = f1_score(y_val_pred_traveling, val_y_traveling, average='macro')
    # F1_score = f1_score(y_val_pred, val_y, average='weighted')
    '''
    '''

    print(set(train["type"]), 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))

    save_result_to_csv(report, F1_score, experiment_id, model_name)

    save_result_to_csv(report_medical, F1_score_medical, experiment_id, model_name + "_medical_")
    save_result_to_csv(report_financial, F1_score_financial, experiment_id, model_name + "_financial_")
    save_result_to_csv(report_traveling, F1_score_traveling, experiment_id, model_name + "_traveling_")
    '''
    '''

    print("%Y-%m-%d %H:%M:%S", time.localtime())

    print(">>>end of train_mlp function...")

    return result


# 训练模型
# 该模型的训练集是餐饮+物流，验证集有三个，分别是医疗 金融 旅游
# train val_1 val_2 val_3为包含原始评论文本的数据，train_x_cnn val_x_1 val_x_2 val_x_3为确定性特征向量
# train_x_fuzzy val_x_fuzzy_1 val_x_fuzzy_2 val_x_fuzzy_3为模糊性特征
def train_model_2(model, train, val, val_1, val_2, val_3, train_x_fuzzy, train_x_cnn, val_x_fuzzy, val_x, val_x_fuzzy_1, val_x_1,
                  val_x_fuzzy_2, val_x_2, val_x_fuzzy_3, val_x_3, y_cols, epoch, experiment_id,
                  batch_size, balanced, model_name, debug=False, folds=1):

    print(">>>in train_model function...")

    # print(mlp_model.summary())
    # print("检查输入数据：")
    # print("train_x_fuzzy.shape = ", train_x_fuzzy.shape)
    # print("test_x.shape = ", test_x.shape)
    # print("val_x.shape = ", val_x.shape)
    # print("y_cols = ", y_cols)

    experiment_id = "fusion_model_" + experiment_id

    F1_scores = 0
    F1_score = 0
    result = {}

    train_y = train["label"]
    val_y = val["label"]
    val_y_1 = val_1["label"]
    val_y_2 = val_2["label"]
    val_y_3 = val_3["label"]

    y_val_pred_1 = 0
    y_val_pred_2 = 0
    y_val_pred_3 = 0

    print("train_x_fuzzy.shape:", train_x_fuzzy.shape)
    print("train_x_cnn.shape:", train_x_cnn.shape)
    print("train_y.shape:", train_y.shape)

    print("val_x_fuzzy.shape:", val_x_fuzzy.shape)
    print("val_x.shape:", val_x.shape)
    print("val_y.shape:", val_y.shape)

    print("val_x_fuzzy_1.shape:", val_x_fuzzy_1.shape)
    print("val_x_1.shape:", val_x_1.shape)
    print("val_y_1.shape:", val_y_1.shape)
    print("val_x_fuzzy_2.shape:", val_x_fuzzy_2.shape)
    print("val_x_2.shape:", val_x_2.shape)
    print("val_y_2.shape:", val_y_2.shape)
    print("val_x_fuzzy_3.shape:", val_x_fuzzy_3.shape)
    print("val_x_3.shape:", val_x_3.shape)
    print("val_y_3.shape:", val_y_3.shape)
    # print("train_y = ", train_y)
    # print("val_y = ", val_y)

    for i in range(folds):
        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)
        y_val_onehot_1 = to_categorical(val_y_1)
        y_val_onehot_2 = to_categorical(val_y_2)
        y_val_onehot_3 = to_categorical(val_y_3)

        history = model.fit([train_x_fuzzy, train_x_cnn], y_train_onehot, epochs=epoch, verbose=2,
                            batch_size=batch_size, validation_data=([val_x_fuzzy, val_x], y_val_onehot))

        # 预测验证集和测试集
        # print("val_x = ", val_x)
        print("val_x_fuzzy.shape:", val_x_fuzzy.shape)
        print("val_x.shape:", val_x.shape)
        y_val_pred = model.predict([val_x_fuzzy, val_x])
        print("val_x_fuzzy_medical.shape:", val_x_fuzzy_1.shape)
        print("val_x_medical.shape:", val_x_1.shape)
        y_val_pred_1 = model.predict([val_x_fuzzy_1, val_x_1])
        print("val_x_fuzzy_financial.shape:", val_x_fuzzy_2.shape)
        print("val_x_financial.shape:", val_x_2.shape)
        y_val_pred_2 = model.predict([val_x_fuzzy_2, val_x_2])
        print("val_x_fuzzy_traveling.shape:", val_x_fuzzy_3.shape)
        print("val_x_traveling.shape:", val_x_3.shape)
        y_val_pred_3 = model.predict([val_x_fuzzy_3, val_x_3])

        # 把预测结果保存入文件
        print(">>>将预测的结果存入csv文件中。。。")
        save_predict_result_to_csv(val["review"], y_val_pred)

    y_val_pred = np.argmax(y_val_pred, axis=1)

    y_val_pred_1 = np.argmax(y_val_pred_1, axis=1)
    y_val_pred_2 = np.argmax(y_val_pred_2, axis=1)
    y_val_pred_3 = np.argmax(y_val_pred_3, axis=1)

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("val_y = ", val_y)
    # print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(val_y, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    precision_medical, recall_medical, fscore_medical, support_medical = score(val_y_1, y_val_pred_1)
    precision_financial, recall_financial, fscore_financial, support_financial = score(val_y_2, y_val_pred_2)
    precision_traveling, recall_traveling, fscore_traveling, support_traveling = score(val_y_3, y_val_pred_3)

    report = classification_report(val_y, y_val_pred, digits=4, output_dict=True)

    report_medical = classification_report(val_y_1, y_val_pred_1, digits=4, output_dict=True)
    report_financial = classification_report(val_y_2, y_val_pred_2, digits=4, output_dict=True)
    report_traveling = classification_report(val_y_3, y_val_pred_3, digits=4, output_dict=True)

    F1_score = f1_score(y_val_pred, val_y, average='macro')
    F1_scores += F1_score

    F1_score_medical = f1_score(y_val_pred_1, val_y_1, average='macro')
    F1_score_financial = f1_score(y_val_pred_2, val_y_2, average='macro')
    F1_score_traveling = f1_score(y_val_pred_3, val_y_3, average='macro')

    print(set(train["type"]), 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))

    save_result_to_csv(report, F1_score, experiment_id, model_name)

    save_result_to_csv(report_medical, F1_score_medical, experiment_id, model_name + "_medical_")
    save_result_to_csv(report_financial, F1_score_financial, experiment_id, model_name + "_financial_")
    save_result_to_csv(report_traveling, F1_score_traveling, experiment_id, model_name + "_traveling_")

    return result


# 把预测结果保存到csv
# 0 1 2
def save_predict_result_to_csv(x, y_val_pred):
    print(">>>将val和y_val存入文件...")
    print("y_val_pred's type = ", type(y_val_pred))

    y_val_pred = np.array(y_val_pred)
    # x = np.array(x)

    path = "result/sentence_result/predicts.csv"
    with codecs.open(path, "a", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(y_val_pred)
        f.close()

    path2 = "result/sentence_result/texts.csv"
    with codecs.open(path2, "a", "utf-8") as f2:
        writer = csv.writer(f2)
        for value in x:
            writer.writerow([value])
        f2.close()


# 把结果保存到csv
# report是classification_report生成的字典结果
def save_result_to_csv(report, f1_score, experiment_id, model_name, F1_score_val_micro):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [experiment_id, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, accuracy, F1_score_val_micro]
    if model_name.startswith("Fusion"):
        path = "result/sentence-202105/nocrossValidation/fusion/20210526/" + model_name + ".csv"
    else:
        path = "result/sentence-202105/nocrossValidation/20210526/" + model_name + ".csv"
    with codecs.open(path, "a", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


# textCNN
def create_textCNN_model(cnn_maxlen, dict_length, cnn_filter, embedding_matrix, metrics_name, filter_sizes='2,3,4,5'):
    print("开始构建textCNN模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = keras.Input(shape=(cnn_maxlen,), name='input_data')
        embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)
        embed = keras.layers.Embedding(dict_length, 300,
                                       embeddings_initializer=embed_initer,
                                       input_length=cnn_maxlen,
                                       weights=[embedding_matrix],
                                       name='embedding')(inputs)
        # single channel. If using real embedding, you can set one static
        embed = keras.layers.Reshape((cnn_maxlen, 300, 1), name='add_channel')(embed)

        pool_outputs = []
        for filter_size in list(map(int, filter_sizes.split(','))):
            filter_shape = (filter_size, 300)
            conv = keras.layers.Conv2D(cnn_filter, filter_shape, strides=(1, 1), padding='valid',
                                       data_format='channels_last', activation='relu',
                                       kernel_initializer='glorot_normal',
                                       bias_initializer=keras.initializers.constant(0.1),
                                       name='convolution_{:d}'.format(filter_size))(embed)
            max_pool_shape = (cnn_maxlen - filter_size + 1, 1)
            pool = keras.layers.MaxPool2D(pool_size=max_pool_shape,
                                          strides=(1, 1), padding='valid',
                                          data_format='channels_last',
                                          name='max_pooling_{:d}'.format(filter_size))(conv)
            pool_outputs.append(pool)

        pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')
        pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)
        pool_outputs = keras.layers.Dropout(0.5, name='dropout')(pool_outputs)

        outputs = keras.layers.Dense(3, activation='softmax',
                                     kernel_initializer='glorot_normal',
                                     bias_initializer=keras.initializers.constant(0.1),
                                     kernel_regularizer=keras.regularizers.l2(0.01),
                                     bias_regularizer=keras.regularizers.l2(0.01),
                                     name='dense')(pool_outputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[metrics_name])
    return model


# attention cnn
def create_attention_cnn_model(cnn_maxlen, dict_length, cnn_filter, embedding_matrix, window_size, metrics_name):
    print("开始构建attention_cnn模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，300是词向量的维度，512是每个input的长度
        embed = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix])(inputs)
        attention_layer = Attention()([embed, embed])


# cnn
# def create_cnn_model(cnn_maxlen, dict_length, filters, embedding_matrix):
def create_cnn_model(cnn_maxlen, dict_length, cnn_filter, embedding_matrix, window_size, metrics_name):
    print("开始构建CNN模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_cnn = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，300是词向量的维度，512是每个input的长度
        x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix])(inputs_cnn)
        # x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=True)(inputs_cnn)
        # x_cnn = Embedding(input_dim=dict_length, output_dim=200, name='embedding_cnn')(inputs_cnn)
        x_cnn = Conv1D(cnn_filter, window_size, padding='same', activation='relu', name='conv')(x_cnn)
        # x_cnn = BatchNormalization(axis=chanDim)(x_cnn)
        x_cnn = MaxPool1D(name='pool')(x_cnn)
        x_cnn = Flatten(name='flatten')(x_cnn)
        # x = Dense(32, activation='relu', name='dense3')(x_cnn)
        # x = Dropout(dropout, name="dropout1")(x)
        x = Dense(3, activation='softmax', name='softmax')(x_cnn)
        cnn_model = Model(inputs=inputs_cnn, outputs=x, name='cnn_model')
        print(cnn_model.summary())
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        cnn_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    # return our model
    return cnn_model


# MLP
def create_mlp_model(maxlen, dict_length, embedding_matrix):
    print("开始构建mlp模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs = Input(shape=(maxlen,), name="inputs")
        x = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix])(inputs)
        x = Flatten(name='flatten')(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5, name="dropout1")(x)
        x = Dense(3, activation='softmax', name='softmax')(x)
        model = Model(inputs=inputs, outputs=x)
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    return model


# CNN-BiGRU
def create_cnn_bigru_model(cnn_maxlen, dict_length, filter, embedding_matrix, window_size, dim):
    print("开始构建cnn_bigru模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_cnn = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，300是词向量的维度，512是每个input的长度
        x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix])(inputs_cnn)
        # x_cnn = Embedding(input_dim=dict_length, output_dim=200, name='embedding_cnn')(inputs_cnn)
        x_cnn = Conv1D(filter, window_size, activation='relu', name='conv')(x_cnn)
        # x_cnn = BatchNormalization(axis=chanDim)(x_cnn)
        x_cnn = MaxPool1D(name='pool')(x_cnn)
        bi_gru = Bidirectional(GRU(dim, name="lstm_1"))(x_cnn)

        # x = Dropout(0.4, name='dropout')(x)
        x = Dense(3, activation='softmax', name='softmax')(bi_gru)

        model = Model(inputs=inputs_cnn, outputs=x)
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    return model


# CNN-BiLSTM
def create_cnn_bilstm_model(cnn_maxlen, dict_length, embedding_matrix):
    print("开始构建cnn_bilstm模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_cnn = Input(shape=(cnn_maxlen,), name="input_cnn")  # dict_length是词典长度，300是词向量的维度，512是每个input的长度
        x_cnn = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix])(inputs_cnn)
        # x_cnn = Embedding(input_dim=dict_length, output_dim=200, name='embedding_cnn')(inputs_cnn)
        x_cnn = Conv1D(128, 3, padding='same', activation='relu', name='conv')(x_cnn)
        # x_cnn = BatchNormalization(axis=chanDim)(x_cnn)
        x_cnn = MaxPool1D(pool_size=2, name='pool')(x_cnn)

        # x_lstm = Bidirectional(LSTM(64, return_sequences=True))(x_cnn)
        x_lstm = Dropout(0.5)(Bidirectional(LSTM(64, activation='tanh'))(x_cnn))
        # x_lstm = LSTM(64)(embedding)
        x_lstm = Dense(3, activation='softmax', name='FC2')(x_lstm)

        model = Model(inputs=inputs_cnn, outputs=x_lstm)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["mae"])

    return model


# train cnn
def train_cnn_model(model, train, val, train_x, test_x, val_x, epoch,
                    experiment_id, batch_size, learning_rate, model_name, debug=False, folds=1):
    print(">>>in train_cnn_model function。。。")

    experiment_id = "cnn_model_" + experiment_id

    # print("train = ", train)
    # print("val = ", val)
    # print("train_x = ", train_x)
    print("train_x.shape = ", train_x.shape)
    # print("test_x = ", test_x)
    # print("val_x = ", val_x)

    adam = optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    F1_scores = 0
    F1_score = 0
    result = {}

    train_y = train['label']
    val_y = val['label']
    y_val_pred = 0
    y_test_pred = 0

    print("train_y.shape:", train_y.shape)
    print("val_x.shape:", val_x.shape)
    print("val_y.shape:", val_y.shape)

    for i in range(folds):
        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)
        # print("y_train_onehot = ", y_train_onehot)
        # print("y_val_onehot = ", y_val_onehot)
        history = model.fit(train_x, y_train_onehot, epochs=epoch, verbose=2,
                            batch_size=batch_size, validation_data=(val_x, y_val_onehot))
                            # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)])

    # 预测验证集和测试集
    y_val_pred = model.predict(val_x)
    y_test_pred += model.predict(test_x)

    y_val_pred = np.argmax(y_val_pred, axis=1)

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(val_y, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    report = classification_report(val_y, y_val_pred, digits=4, output_dict=True)

    print(report)

    F1_score = f1_score(y_val_pred, val_y, average='macro')
    # F1_score = f1_score(y_val_pred, val_y, average='weighted')

    print('f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
    y_test_pred = np.argmax(y_test_pred, axis=1)

    save_result_to_csv(report, F1_score, experiment_id, model_name)

    print(">>>end of train_cnn_model function in featureFusion.py。。。")

    return result

# 融合模型的训练函数，不要交叉验证
def train_fusion_model_no_cross_validation(model, train, val, dealed_train, dealed_val, dealed_train_fuzzy, dealed_val_fuzzy, epoch,
                                           experiment_id, batch_size, model_name, dealed_tests, dealed_tests_fuzzy, data_tests, data_test_names):
    print("正在训练融合模型。。。")

    experiment_id = model_name + "_" + experiment_id
    print("dealed_train.shape = ", dealed_train.shape)
    print("train'length = ", len(train))
    print("dealed_train_fuzzy'length = ", len(dealed_train_fuzzy))
    if len(dealed_train) != len(train) or len(dealed_train_fuzzy) != len(train):
        print("dealed_data dealed_fuzzy长度不一致！！！出错辣！！！")
        sys.exit(-1)

    y_train = train['label']
    y_train = np.array(y_train)
    y_train_onehot = to_categorical(y_train)
    y_val = val['label']
    y_val = np.array(y_val)
    y_val_onehot = to_categorical(y_val)


# 融合模型的训练函数，没有交叉验证
def train_fusion_model_no_cross(model, train, val, dealed_train, dealed_val, dealed_fuzzy_train, dealed_fuzzy_val, epoch, experiment_id, batch_size, model_name, dealed_tests, data_tests, data_test_names, dealed_tests_fuzzy):
    print(">>>in train_fusion_model_no_cross function。。。")
    experiment_id = model_name + "_" + experiment_id

    y_train = train['label']
    y_train = np.array(y_train)
    y_train_onehot = to_categorical(y_train)
    y_val = val['label']
    y_val = np.array(y_val)
    y_val_onehot = to_categorical(y_val)

    # 早停法，如果val_acc没有提高0.0001就停止
    earlystop_callback =EarlyStopping(monitor='val_loss', mode='min', patience=3)
    # earlystop_callback =EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=1)
    model.fit([dealed_train, dealed_fuzzy_train], y_train_onehot, epochs=epoch, verbose=2, batch_size=batch_size, callbacks=[earlystop_callback], validation_data=([dealed_val, dealed_fuzzy_val], y_val_onehot))

    # 预测验证集
    y_val_pred = model.predict([dealed_val, dealed_fuzzy_val])
    y_val_pred = np.argmax(y_val_pred, axis=1)
    report_val = classification_report(y_val, y_val_pred, digits=4, output_dict=True)
    F1_score_val = f1_score(y_val, y_val_pred, average='macro')
    F1_score_val_micro = f1_score(y_val, y_val_pred, average='micro')
    print('F1_score_val:', F1_score_val, 'ACC_score:', accuracy_score(y_val, y_val_pred))
    save_result_to_csv(report_val, F1_score_val, experiment_id, model_name, F1_score_val_micro)

    # 预测测试集
    '''
    test_length = len(dealed_tests)
    for i in range(test_length):
        dealed_test = dealed_tests[i]
        dealed_test_fuzzy = dealed_tests_fuzzy[i]
        test = data_tests[i]
        name = data_test_names[i]
        # 获取Y
        y_test = test['label']
        y_test_pred = model.predict([dealed_test, dealed_test_fuzzy])
        y_test_pred = np.argmax(y_test_pred, axis=1)
        report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
        F1_score_test = f1_score(y_test, y_test_pred, average='macro')
        print('F1_score_test:', F1_score_test, 'ACC_score:', accuracy_score(y_test, y_test_pred))
        experiment_id = experiment_id + "_" + name
        save_result_to_csv(report, F1_score_test, experiment_id, model_name)
    '''

    return y_val_pred


# 融合模型的训练函数
def train_fusion_model(model, data, dealed_data, dealed_fuzzy, epoch, experiment_id, batch_size, model_name, dealed_tests, data_tests, data_test_names):
    print("正在训练融合模型。。。")

    experiment_id = model_name + "_" + experiment_id

    print("dealed_train.shape = ", dealed_data.shape)
    result = {}
    print("dealed_data'length = ", len(dealed_data))
    print("dealed_fuzzy'length = ", len(dealed_fuzzy))
    print("data'length = ", len(data))
    if len(dealed_data) != len(dealed_fuzzy) or len(dealed_data) != len(data):
        print("dealed_data dealed_fuzzy长度不一致！！！出错辣！！！")
        sys.exit(-1)
    y = data['label']
    print("y's type = ", type(y))
    y = np.array(y)
    kf = KFold(n_splits=10)
    current_k = 0
    for train_index, validation_index in kf.split(data):
        print("正在进行第", current_k, "轮交叉验证...")
        current_k += 1
        train_x = dealed_data[train_index]
        print("train_x.shape = ", train_x.shape)
        train_fuzzy_x = dealed_fuzzy[train_index]
        print("train_fuzzy_x.shape = ", train_fuzzy_x.shape)
        train_y = y[train_index]
        print("train_y.shape = ", train_y.shape)
        train_y_onehot = to_categorical(train_y)
        val_x = dealed_data[validation_index]
        print("val_x.shape = ", val_x.shape)
        val_fuzzy_x = dealed_fuzzy[validation_index]
        print("val_fuzzy_x.shape = ", val_fuzzy_x.shape)
        val_y = y[validation_index]
        print("val_y.shape = ", val_y.shape)
        val_y_onehot = to_categorical(val_y)
        model.fit([train_x, train_fuzzy_x], train_y_onehot, epochs=epoch, verbose=2, batch_size=batch_size, validation_data=([val_x, val_fuzzy_x], val_y_onehot))
        y_val_pred = model.predict([val_x, val_fuzzy_x])
        y_val_pred = np.argmax(y_val_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        # print("y_val_pred = ", list(y_val_pred))
        precision, recall, fscore, support = score(val_y, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        report = classification_report(val_y, y_val_pred, digits=4, output_dict=True)

        print("report:", report)

        F1_score = f1_score(y_val_pred, val_y, average='macro')

        print('f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))

        save_result_to_csv(report, F1_score, experiment_id, model_name)

    # 预测测试集（其他领域数据）
    test_length = len(dealed_tests)
    for i in range(test_length):
        dealed_test = dealed_tests[i]
        test = data_tests[i]
        name = data_test_names[i]
        # 获取Y
        y_test = test['label']
        y_test_pred = model.predict(dealed_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
        F1_score_test = f1_score(y_test, y_test_pred, average='macro')
        print('F1_score_test:', F1_score_test, 'ACC_score:', accuracy_score(y_test, y_test_pred))
        experiment_id = experiment_id + "_" + name
        save_result_to_csv(report, F1_score_test, experiment_id, model_name)

    print(">>>end of train_fusion_model function in featureFusion_sentence.py。。。")

    return result


'''
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return
'''


# 通用的模型训练函数，不适用交叉验证
def train_all_model_no_cross_validation(model, train, val, dealed_train, dealed_val, epoch, experiment_id, batch_size, model_name, dealed_tests, data_tests, data_test_names):
    print(">>>in train_all_model_cross_validation function。。。")
    experiment_id = model_name + "_" + experiment_id

    y_train = train['label']
    y_train = np.array(y_train)
    y_train_onehot = to_categorical(y_train)
    y_val = val['label']
    y_val = np.array(y_val)
    y_val_onehot = to_categorical(y_val)

    # 早停法，如果val_acc没有提高0.0001就停止
    earlystop_callback =EarlyStopping(monitor='val_loss', mode='min', patience=3)
    # earlystop_callback =EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=1)
    model.fit(dealed_train, y_train_onehot, epochs=epoch, verbose=2, batch_size=batch_size, callbacks=[earlystop_callback], validation_data=(dealed_val, y_val_onehot))

    # 预测验证集
    y_val_pred = model.predict(dealed_val)
    y_val_pred = np.argmax(y_val_pred, axis=1)
    report_val = classification_report(y_val, y_val_pred, digits=4, output_dict=True)
    F1_score_val = f1_score(y_val, y_val_pred, average='macro')
    print('F1_score_val:', F1_score_val, 'ACC_score:', accuracy_score(y_val, y_val_pred))
    save_result_to_csv(report_val, F1_score_val, experiment_id, model_name)

    # 预测测试集
    test_length = len(dealed_tests)
    for i in range(test_length):
        dealed_test = dealed_tests[i]
        test = data_tests[i]
        name = data_test_names[i]
        # 获取Y
        y_test = test['label']
        y_test_pred = model.predict(dealed_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
        F1_score_test = f1_score(y_test, y_test_pred, average='macro')
        print('F1_score_test:', F1_score_test, 'ACC_score:', accuracy_score(y_test, y_test_pred))
        experiment_id = experiment_id + "_" + name
        save_result_to_csv(report, F1_score_test, experiment_id, model_name)

    # 保存验证集预测结果,val, dealed_fuzzy_val, y_val, y_val_pred
    return y_val_pred


# 保存验证集预测结果
def save_validation_result_to_csv(val, dealed_fuzzy_val, y_val_pred, y_val_pred_fusion):
    print("正在保存验证集预测结果...")
    texts = np.array(val['review'])
    dealed_fuzzy_val = dealed_fuzzy_val[:, 0]
    y_val = np.array(val['label'])
    if len(texts) != len(dealed_fuzzy_val) or len(texts) != len(y_val_pred):
        print("验证集文本和label的长度不一致！！！出错辣！！！")
        sys.exit(-3)
    length = len(texts)
    print("length = ", length)
    path = 'result/sentence-202105/nocrossValidation/predicted.csv'
    with codecs.open(path, "a", "utf-8") as f:
        writer = csv.writer(f)
        for i in range(length):
            current_fuzzy_neg = dealed_fuzzy_val[i][0]
            current_fuzzy_neu = dealed_fuzzy_val[i][1]
            current_fuzzy_pos = dealed_fuzzy_val[i][2]
            data = [texts[i], y_val[i], y_val_pred[i], current_fuzzy_neg, current_fuzzy_neu, current_fuzzy_pos, y_val_pred_fusion[i]]
            print("data = ", data)
            writer.writerow(data)
        f.close()


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)


# 通用的模型训练函数
def train_all_model_cross_validation(model, data, dealed_data, epoch, experiment_id, batch_size, model_name, dealed_tests, data_tests, data_test_names):
    print(">>>in train_all_model_cross_validation function。。。")
    experiment_id = model_name + "_" + experiment_id
    print("dealed_train.shape = ", dealed_data.shape)
    result = {}
    if len(dealed_data) != len(data):
        print("dealed_data data长度不一致！！！出错辣！！！")
        sys.exit(-1)
    y = data['label']
    y = np.array(y)
    kf = KFold(n_splits=4)
    current_k = 0
    for train_index, validation_index in kf.split(data):
        print("正在进行第", current_k, "轮交叉验证...")
        current_k += 1
        train_x = dealed_data[train_index]
        print("train_x.shape = ", train_x.shape)
        train_y = y[train_index]
        print("train_y.shape = ", train_y.shape)
        train_y_onehot = to_categorical(train_y)
        val_x = dealed_data[validation_index]
        print("val_x.shape = ", val_x.shape)
        val_y = y[validation_index]
        print("val_y.shape = ", val_y.shape)
        val_y_onehot = to_categorical(val_y)
        model.fit(train_x, train_y_onehot, epochs=epoch, verbose=2, batch_size=batch_size, validation_data=(val_x, val_y_onehot))
        y_val_pred = model.predict(val_x)
        y_val_pred = np.argmax(y_val_pred, axis=1)

        # 准确率：在所有预测为正的样本中，确实为正的比例
        # 召回率：本身为正的样本中，被预测为正的比例
        # print("y_val_pred = ", list(y_val_pred))
        precision, recall, fscore, support = score(val_y, y_val_pred)
        print("precision = ", precision)
        print("recall = ", recall)
        print("fscore = ", fscore)
        print("support = ", support)

        report = classification_report(val_y, y_val_pred, digits=4, output_dict=True)

        print("report:", report)

        F1_score = f1_score(val_y, y_val_pred, average='macro')

        print('f1_score:', F1_score, 'ACC_score:', accuracy_score(val_y, y_val_pred))

        save_result_to_csv(report, F1_score, experiment_id, model_name)

    # 预测测试集（其他领域数据）
    test_length = len(dealed_tests)
    for i in range(test_length):
        dealed_test = dealed_tests[i]
        test = data_tests[i]
        name = data_test_names[i]
        # 获取Y
        y_test = test['label']
        y_test_pred = model.predict(dealed_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        report = classification_report(y_test, y_test_pred, digits=4, output_dict=True)
        F1_score_test = f1_score(y_test, y_test_pred, average='macro')
        print('F1_score_test:', F1_score_test, 'ACC_score:', accuracy_score(y_test, y_test_pred))
        experiment_id = experiment_id + "_" + name
        save_result_to_csv(report, F1_score_test, experiment_id, model_name)

    print(">>>end of train_all_model_cross_validation function in featureFusion_sentence.py。。。")

    return result


# 通用的模型训练函数
def train_all_model(model, train, val, train_x, val_x, epoch, experiment_id, batch_size, model_name):
    print(">>>in train_all_model function。。。")

    experiment_id = model_name + "_" + experiment_id

    print("train_x.shape = ", train_x.shape)
    result = {}

    train_y = train['label']
    val_y = val['label']

    print("train_y.shape:", train_y.shape)
    print("val_x.shape:", val_x.shape)
    print("val_y.shape:", val_y.shape)

    y_train_onehot = to_categorical(train_y)
    y_val_onehot = to_categorical(val_y)
    history = model.fit(train_x, y_train_onehot, epochs=epoch, verbose=2,
                        batch_size=batch_size, validation_data=(val_x, y_val_onehot))

    # 预测验证集和测试集
    y_val_pred = model.predict(val_x)
    y_val_pred = np.argmax(y_val_pred, axis=1)

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(val_y, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    report = classification_report(val_y, y_val_pred, digits=4, output_dict=True)

    print("report:", report)

    F1_score = f1_score(y_val_pred, val_y, average='macro')

    print('f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))

    save_result_to_csv(report, F1_score, experiment_id, model_name)

    print(">>>end of train_all_model function in featureFusion_sentence.py。。。")

    return result


# train cnn
def train_cnn_model_2(model, train, val, val_medical, val_financial, val_traveling, train_x, val_x,
                      val_x_medical, val_x_financial, val_x_traveling, epoch,
                    experiment_id, batch_size, learning_rate, model_name, debug=False, folds=1):
    print(">>>in train_cnn_model function。。。")

    experiment_id = "cnn_model_" + experiment_id

    # print("train = ", train)
    # print("val = ", val)
    # print("train_x = ", train_x)
    print("train_x.shape = ", train_x.shape)
    # print("test_x = ", test_x)
    # print("val_x = ", val_x)

    adam = optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])

    F1_scores = 0
    F1_score = 0
    result = {}

    train_y = train['label']
    val_y = val['label']
    val_y_medical = val_medical['label']
    val_y_financial = val_financial['label']
    val_y_traveling = val_traveling['label']
    y_val_pred = 0

    print("train_y.shape:", train_y.shape)
    print("val_x.shape:", val_x.shape)
    print("val_y.shape:", val_y.shape)

    for i in range(folds):
        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)
        # print("y_train_onehot = ", y_train_onehot)
        # print("y_val_onehot = ", y_val_onehot)
        history = model.fit(train_x, y_train_onehot, epochs=epoch, verbose=2,
                            batch_size=batch_size, validation_data=(val_x, y_val_onehot))
        # callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)])

    # 预测验证集和测试集
    y_val_pred = model.predict(val_x)
    y_val_pred_medical = model.predict(val_x_medical)
    y_val_pred_financial = model.predict(val_x_financial)
    y_val_pred_traveling = model.predict(val_x_traveling)

    y_val_pred = np.argmax(y_val_pred, axis=1)
    y_val_pred_medical = np.argmax(y_val_pred_medical, axis=1)
    y_val_pred_financial = np.argmax(y_val_pred_financial, axis=1)
    y_val_pred_traveling = np.argmax(y_val_pred_traveling, axis=1)

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(val_y, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    report = classification_report(val_y, y_val_pred, digits=4, output_dict=True)
    report_medical = classification_report(val_y_medical, y_val_pred_medical, digits=4, output_dict=True)
    report_financial = classification_report(val_y_financial, y_val_pred_financial, digits=4, output_dict=True)
    report_traveling = classification_report(val_y_traveling, y_val_pred_traveling, digits=4, output_dict=True)

    print("report:", report)
    print("report_medical:", report_medical)
    print("report_financial:", report_financial)
    print("report_traveling:", report_traveling)

    F1_score = f1_score(y_val_pred, val_y, average='macro')
    F1_score_medical = f1_score(val_y_medical, y_val_pred_medical, average='macro')
    F1_score_financial = f1_score(val_y_financial, y_val_pred_financial, average='macro')
    F1_score_traveling = f1_score(val_y_traveling, y_val_pred_traveling, average='macro')

    print('f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))

    save_result_to_csv(report, F1_score, experiment_id, model_name)
    save_result_to_csv(report_medical, F1_score_medical, experiment_id, model_name + "_medical")
    save_result_to_csv(report_financial, F1_score_financial, experiment_id, model_name + "_financial")
    save_result_to_csv(report_traveling, F1_score_traveling, experiment_id, model_name + "_traveling")

    print(">>>end of train_cnn_model function in featureFusion.py。。。")

    return result


# 生成预训练词向量,size-
def load_word2vec(word_index):
    print(">>>in load_word2vec function of featureFusion.py...")
    print("word_index's lengh = ", len(word_index))
    # f = open(config_sentence.pre_word_embedding_3, "r", encoding="utf-8")
    # f = open(config_sentence.pre_word_embedding_2, "r", encoding="utf-8")
    f = open(config_sentence.pre_word_embedding, "r", encoding="utf-8")
    length, dimension = f.readline().split()  # 预训练词向量的单词数和词向量维度
    dimension = int(dimension)
    print("length = ", length, ", dimension = ", dimension)

    # 创建词向量索引字典
    embeddings_index = {}

    print(">>>读取预训练词向量ing。。。")

    for line in f:
        # print("line = ", line)
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
        # print(word, ":", coefs)
    f.close()

    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    # 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))
    # 遍历词汇表中的每一项
    for word, i in word_index.items():
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        # print("embedding_vector = ", embedding_vector)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        # print(word, ":", embedding_vector)
        # print("vector's length = ", len(embedding_vector))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # print("embedding_matrix = ", embedding_matrix)
    print(">>>end of load_word2vec function in featureFusion.py...")

    return embedding_matrix


# fasttext model
def fasttext_model(fea_dict, maxlen):
    inputs_fasttext = Input(shape=(maxlen,), name='fasttext_input')
    x_fasttext = Embedding(input_dim=len(fea_dict), output_dim=200, name='embedding_fasttext')(inputs_fasttext)
    x_fasttext = GlobalAveragePooling1D(name='GlobalAveragePooling1D_fasttext')(x_fasttext)
    x_fasttext = Dropout(0.3)(x_fasttext)
    x_fasttext = Dense(3, activation='softmax', name='softmax')(x_fasttext)

    model = Model(inputs=inputs_fasttext, outputs=x_fasttext, name='fasttext_model')

    print(model.summary())

    return model


# train fasttext model
def train_fasttext_model(model, train, val, train_x, test_x, val_x, epoch, debug=False, folds=1):
    print(">>>in train_fasttext_model function。。。")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    F1_scores = 0
    F1_score = 0
    result = {}

    train_y = train["label"]
    val_y = val["label"]
    y_val_pred = 0
    y_test_pred = 0

    print("train_y.shape:", train_y.shape)
    print("val_x.shape:", val_x.shape)
    print("val_y.shape:", val_y.shape)

    for i in range(folds):
        y_train_onehot = to_categorical(train_y)
        y_val_onehot = to_categorical(val_y)
        print("train_x.shape = ", train_x.shape)
        print("y_train_onehot.shape = ", y_train_onehot.shape)
        print("val_x.shape = ", val_x.shape)
        print("y_val_onehot.shape = ", y_val_onehot.shape)
        history = model.fit(train_x, y_train_onehot, epochs=epoch,
                            batch_size=64, validation_data=(val_x, y_val_onehot))

        # 预测验证集和测试集
        y_val_pred = model.predict(val_x)
        y_test_pred += model.predict(test_x)

    y_val_pred = np.argmax(y_val_pred, axis=1)

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(val_y, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    print(classification_report(val_y, y_val_pred, digits=4))

    F1_score = f1_score(y_val_pred, val_y, average='weighted')
    F1_scores += F1_score

    print('f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
    y_test_pred = np.argmax(y_test_pred, axis=1)
    print("result:", result)

    print(">>>end of train_fasttext_model function in featureFusion.py。。。")

    return result


# lstm+attention
def create_attention_lstm_model(maxlen, dict_length, embedding_matrix, dim):
    print("开始构建新的attention_lstm融合模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_determinacy_0 = Input(shape=(maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        inputs_determinacy = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=False)(inputs_determinacy_0)

        x_lstm = Bidirectional(LSTM(dim, return_sequences=True))(inputs_determinacy)
        attention_layer = Attention()([x_lstm, x_lstm])
        pooling_out1 = GlobalMaxPooling1D()(x_lstm)
        pooling_out2 = GlobalMaxPooling1D()(attention_layer)
        merge = Concatenate()([pooling_out1, pooling_out2])

        x = Dropout(0.5, name="dropout2")(merge)
        x = Dense(3, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=inputs_determinacy_0, outputs=x, name='fusion_lstm_model')
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    print(fusion_model.summary())

    return fusion_model


# fusion+lstm+attention
def create_fusion_attention_lstm_model(maxlen, dict_length, embedding_matrix, dim):
    print("开始构建新的fusion_attention_lstm融合模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_determinacy_0 = Input(shape=(maxlen,), name="input_cnn")  # dict_length是词典长度，128是词向量的维度，512是每个input的长度
        inputs_determinacy = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix], trainable=False)(inputs_determinacy_0)
        inputs_fuzzy = Input(shape=(maxlen, 3,), name='input_fuzzy')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
        inputs = concatenate([inputs_determinacy, inputs_fuzzy], axis=-1)

        x_lstm = Bidirectional(LSTM(dim, return_sequences=True))(inputs)
        attention_layer = Attention()([x_lstm, x_lstm])
        pooling_out1 = GlobalMaxPooling1D()(x_lstm)
        pooling_out2 = GlobalMaxPooling1D()(attention_layer)
        merge = Concatenate()([pooling_out1, pooling_out2])

        x = Dropout(0.5, name="dropout2")(merge)
        x = Dense(3, activation='softmax', name='softmax')(x)

        fusion_model = Model(inputs=[inputs_determinacy_0, inputs_fuzzy], outputs=x, name='fusion_lstm_model')
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        fusion_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

    print(fusion_model.summary())

    return fusion_model


# lstm模型
def create_lstm_model(maxlen, dict_length, embedding_matrix, dim):
    '''
    def attention_3d_block(inputs):
        #input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Dense(int(maxlen/2), activation='softmax')(a)
        a_probs = Permute((2, 1), name='attention_vec')(a)
        #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
        output_attention_mul = multiply(inputs, a_probs, name='attention_mul')
        return output_attention_mul
    '''

    print("开始构建LSTM模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_lstm = Input(shape=(maxlen,), name='inputs_lstm')

        embedding = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix])(inputs_lstm)
        merge=Dropout(0.5)(LSTM(units=dim, activation='tanh')(embedding))
        '''
        x_lstm = Bidirectional(LSTM(units=dim, return_sequences=True, kernel_regularizer=regularizers.l1(0.01)), name='bilstm')(embedding)
        attention_layer = Attention()([x_lstm, x_lstm])
        pooling_out1 = GlobalMaxPooling1D()(x_lstm)
        pooling_out2 = GlobalMaxPooling1D()(attention_layer)
        merge = Concatenate()([pooling_out1, pooling_out2])
        '''
        '''
        attention_mul = attention_3d_block(x_lstm)
        attention_flatten = Flatten()(attention_mul)
        '''
        drop2 = Dropout(0.5)(merge)

        x = Dense(3, activation='softmax', name='FC2')(drop2)

        model = Model(inputs=inputs_lstm, outputs=x, name='lstm_model')

        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])
        print(model.summary())

    return model


# gru模型
def create_gru_model(maxlen, dict_length, embedding_matrix, dim):
    print("开始构建GRU模型。。。")
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        inputs_gru = Input(shape=(maxlen,), name='inputs_lstm')

        # embedding = Embedding(input_dim=dict_length, output_dim=200, name='embedding_cnn')(inputs_lstm)
        embedding = Embedding(input_dim=dict_length, output_dim=300, name='embedding_cnn', weights=[embedding_matrix])(inputs_gru)

        # bi_gru = Bidirectional(GRU(dim, name="gru_1"))(embedding)
        bi_gru = GRU(dim, name='gru')(embedding)
        # x_lstm = LSTM(64)(embedding)
        # x_gru = Dense(32, activation='relu', name='FC1')(bi_gru)
        x_gru = Dropout(0.5)(bi_gru)
        x_gru = Dense(3, activation='softmax', name='FC2')(x_gru)

        model = Model(inputs=inputs_gru, outputs=x_gru, name='gru_model')
        adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['mae'])

        print(model.summary())

    return model


# 全连接神经网络，测试情感向量
def fnn_model(input_shape, input_shape2):
    # define our MLP network
    inputs_fnn = Input(shape=(input_shape,), name='input_fnn')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
    x_fnn = Dense(4, activation='relu', name='dense1_fnn')(inputs_fnn)
    x_fnn = Dense(2, activation='linear', name='dense2_fnn')(x_fnn)

    # x = Dense(2, activation='softmax', name='softmax')(x_fnn)

    inputs_fnn2 = Input(shape=(input_shape2,), name='input_fnn2')  # 此处的维度512是根据语料库计算出来的，后期用变量代替
    x_fnn2 = Dense(4, activation='relu', name='dense1_fnn2')(inputs_fnn2)
    x_fnn2 = Dense(3, activation='linear', name='dense2_fnn2')(x_fnn2)

    # 融合两个输入
    x_concatenate = concatenate([x_fnn, x_fnn2], name='fusion')

    x = Dense(32, activation='relu', name='dense3')(x_concatenate)
    x = Dropout(0.5, name="dropout1")(x)
    x = Dense(2, activation='softmax', name='softmax')(x)

    fusion_model = Model(inputs=[inputs_fnn, inputs_fnn2], outputs=x, name='fusion_model')

    print(fusion_model.summary())

    # return our model
    return fusion_model


# 读取模型并预测
def load_predict(model, dealed_test_fuzzy, dealed_test, texts):
    print(">>>in load_predict function...")
    predicted_result = model.predict([dealed_test_fuzzy, dealed_test])
    # print(predicted_result)
    # print("predicted_result's type = ", type(predicted_result))
    predicted_result = list(predicted_result)
    texts = list(texts)

    length = len(texts)
    data = []
    for i in range(length):
        # print("result:" + list(predicted_result[i]) + ", TEXT:" + test[i])
        # print("predicted = ", predicted_result[i])
        current_predicted = list(predicted_result[i])
        # print("text = ", texts[i])
        current = [current_predicted[0], current_predicted[1], current_predicted[2], texts[i]]
        print("current = ", current)
        data.append(current)

    # 写入csv文件
    with codecs.open("result/predicted.csv", "a", "utf-8_sig") as f:
        writer = csv.writer(f)
        writer.writerows(data)
        f.close()

    print(">>>end of load_predict function...")


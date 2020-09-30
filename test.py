# -*- coding:utf-8 -*-

import numpy
import KMeansCluster as km
import keras
import keras.backend as K
import tensorflow as tf
import jieba
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from keras import Input, Model

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report

from dataProcess_sentence import adaption_predict
import dataProcess_sentence as dp_s

import codecs
import csv

from collections import Counter
import config
import KMeansCluster
import fuzzySystem


def compute_test():
    array = [1, 2, 2, 8, 5]
    print(array)
    array = numpy.array(array)

    print("array:", array)

    array_inverse = 1 / array
    print("array_inverse:", array_inverse)

    array_inverse_sum = numpy.sum(array_inverse)
    print("array_inverse_sum:", array_inverse_sum)

    result = 1 / (array * array_inverse_sum)
    print("result:", result)


def dict_test():
    dict1 = {"张三": 25, "李四": 31}
    print(dict1.get("李四"))
    print(dict1.get("李四0", 1))


def concatenate_test():
    tt1 = K.variable(numpy.array([[[0, 22], [29, 38]], [[49, 33], [5, 3]], [[8, 8], [9, 9]]]))
    tt2 = K.variable(numpy.array([[[55, 47], [88, 48]], [[28, 10], [15, 51]], [[5, 5], [6, 6]]]))

    t1 = K.variable(numpy.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]]))
    t2 = K.variable(numpy.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]]))

    print(type(tt1))

    print("tt1.shape = ", tt1.shape)
    print("tt2.shape = ", tt2.shape)

    dd3 = K.concatenate([tt1, tt2], axis=0)
    print("dd3.shape = ", dd3.shape)


def build_model():
    model = Sequential()
    conv1 = Conv1D(64, 3, activation='relu')
    model.add(conv1)
    pooling1 = MaxPooling1D(5)
    model.add(pooling1)
    conv2 = Conv1D(64, 3, activation='relu')
    model.add(conv2)
    dense = Dense(4, activation='softmax')
    model.add(dense)

    return model


# Keras函数式API，定义多个输入
def build_model2():
    inputs = Input(shape=(10,))
    x = Dense(8, activation='relu')(inputs)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='relu')(x)
    model = Model(inputs, x)


# data test
def data_test():
    # 总人数是1000, 一半是男生
    n = 50
    # 所有的身体指标数据都是标准化数据, 平均值0, 标准差1
    tizhong = np.random.normal(size=n)
    shengao = np.random.normal(size=n)
    nianling = np.random.normal(size=n)

    grade = np.random.normal(size=n)
    talent = np.random.normal(size=n)

    # 性别数据, 前500名学生是男生, 用数字1表示
    gender = np.zeros(n)
    gender[:25] = 1
    # 男生的体重比较重,所以让男生的体重+1
    tizhong[:25] += 1

    # 男生的身高比较高, 所以让男生的升高 + 1
    shengao[:25] += 1

    # 男生的年龄偏小, 所以让男生年龄降低 1
    nianling[:25] -= 1

    data = np.array([tizhong, shengao, nianling]).T
    data2 = np.array([grade, talent]).T

    print("data before = ", np.array([tizhong, shengao, nianling]))
    print("data = ", data)
    print("data's shape = ", data.shape)
    print("data's type = ", type(data))
    print("data2 = ", data2)
    return data, gender, data2


def average_test():
    a = [[3, 1, 5], [1, 5, 1]]
    a = np.array(a)
    print(a)
    print(np.average(a))
    print(np.average(a, axis=0))
    print(np.average(a, axis=1))


# 计算准确率 召回率
def calculate_precision_recall():
    # predicted = [1, 2, 3, 4, 5, 1, 2, 1, 1, 4, 5]
    # y_test = [1, 2, 3, 4, 5, 1, 2, 1, 1, 4, 1]
    trueLabel = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4]
    predicted = [1, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 3, 3, 3, 2, 4, 4, 4]

    precision, recall, fscore, support = score(trueLabel, predicted)

    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    result = classification_report(trueLabel, predicted, digits=4, output_dict=True)

    save_result_to_csv(result, "1.11", "2.1")

    # 只取 accuracy、macro avg、weighted avg三个key的内容
    # accuracy = result.get("accuracy")
    # macro_avg = result.get("macro avg")
    # weighted_avg = result.get("weighted avg")
    # macro_precision = macro_avg.get("precision")
    # macro_recall = macro_avg.get("recall")
    # macro_f1 = macro_avg.get('f1-score')
    # weighted_precision = weighted_avg.get("precision")
    # weighted_recall = weighted_avg.get("recall")
    # weighted_f1 = weighted_avg.get('f1-score')
    # experiment_id = "1.2"
    # f1_score = "1.111"
    # data = [experiment_id, weighted_precision, weighted_recall, weighted_f1, f1_score, macro_precision, macro_recall, macro_f1, f1_score, accuracy]

    # print(type(result))
    # print("result = ", result)
    # print(len(result))
    # result = pd.DataFrame(result).transpose()
    # print(type(result))
    # print(result)


def counter_test():
    sample_statistics = {'location': Counter({-2: 16839, 1: 11252, -1: 1032, 0: 877}), 'service': Counter({1: 13150, -2: 9065, 0: 4254, -1: 3531}), 'price': Counter({1: 10859, -2: 8854, 0: 6876, -1: 3411}), 'environment': Counter({1: 13938, -2: 10533, 0: 2951, -1: 2578}), 'dish': Counter({1: 19688, 0: 6407, -1: 2972, -2: 933}), 'others': Counter({1: 21014, 0: 5737, -1: 2844, -2: 405})}
    values = sample_statistics.values()
    print("values = ", values)
    for value in values:
        print(value.get(-2))
        print(value.values())
        # print("value = ", value)
        # print("value's type = ", type(value))


# 预训练词向量
# vocab是数据预处理后的词汇字典
def pre_word_embedding(vocab):
    f = open(config.pre_word_embedding, "r", encoding="utf-8")
    length, dimension = f.readline().split()  # 预训练词向量的单词数和词向量维度
    dimension = int(dimension)
    print("length = ", length, ", dimension = ", dimension)

    # 创建词向量索引字典
    embeddings_index = {}

    i = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
        print(word, ":", coefs)
        # if i > 10:
        #     break
        # i += 1
    f.close()

    print("len(vocab)", type(len(vocab)))
    print("dimension", type(dimension))

    # 构建词向量矩阵，预训练的词向量中没有出现的词用0向量表示
    # 创建一个0矩阵，这个向量矩阵的大小为（词汇表的长度+1，词向量维度）
    embedding_matrix = np.zeros((len(vocab) + 1, dimension))
    # 遍历词汇表中的每一项
    for word, i in vocab.items():
        # 在词向量索引字典中查询单词word的词向量
        embedding_vector = embeddings_index.get(word)
        print("embedding_vector = ", embedding_vector)
        # 判断查询结果，如果查询结果不为空,用该向量替换0向量矩阵中下标为i的那个向量
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print("embedding_matrix = ", embedding_matrix)

    # 修改模型中嵌入层代码
    embedder = Embedding(len(vocab)+1, dimension, input_length=64, weights=[embedding_matrix], trainable=False)


def data_processs():
    path = "C:\desktop\Research\DataSet\百度点石大赛\data_train.csv"
    columns = ['id', 'type', 'comment', 'label']
    data = pd.read_csv(path, sep='\t', names=columns, encoding='utf-8')

    print("data.shape = ", data.shape)
    data = data.loc[data['type'] == str('物流快递')]
    print("data = ", data)
    print("data.shape = ", data.shape)
    print(data.columns.values.tolist())
    print(data['id'].shape)
    print("data['id'].shape = ", data['id'].shape)
    print(set(data['type']))
    print("set(data['type']) = ", set(data['type']))
    print(set(data['label']))


def write_to_csv():
    header = ["experiment_id", "weighted_precision", "weighted_recall", "weighted_f1_score", "macro_precision", "macro_recall", "macro_f1_score", "f1_score", "acc_score"]

    data = ["1.1", "0.1", "0.2", "0.3", "1.1", "0.1", "0.2", "0.3", "1.1"]
    # data = [["1.1", "0.1", "0.2", "0.3", "1.1", "0.1", "0.2", "0.3", "1.1", "0.1"]]

    with codecs.open("result/result.csv", "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # writer.writerow(data)
        f.close()


def save_result_to_csv(report, f1_score, experiment_id):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [experiment_id, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, f1_score, accuracy]

    with codecs.open("result/result.csv", "a", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


# 测试numpy的均值操作
def numpy_mean():
    a = [[1, 2], [3, 2], [4, 8], [2, 10]]
    # a = np.array(a)
    print("a.mean() = ", np.mean(a))
    print("a.mean(axis=0)", np.mean(a, axis=0))
    print("a.mean(axis=1)", np.mean(a, axis=1))

    np.mean(a)


# tokenizer测试
def tokenizer_texts():
    origin_data, y_cols = dp_s.initData2(2)
    stoplist = dp_s.getStopList()
    texts = dp_s.processDataToTexts(origin_data, stoplist)
    print("texts = ", texts)
    dict_length = 500
    maxlen = 52
    tokenizer = Tokenizer(num_words=dict_length)
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index
    print("word_index = ", word_index)
    print("word_index.length = ", len(word_index))
    print("word_index.type = ", type(word_index))

    texts_test = [['医生', '态度', '挺', '好', '最', '重要', '没有', '花太多']]

    data_w = tokenizer.texts_to_sequences(texts_test)
    data_T = sequence.pad_sequences(data_w, maxlen=maxlen)
    print("data_T = ", list(data_T))


import matplotlib.pyplot as plt
from pylab import *
import matplotlib.ticker as ticker
mpl.rcParams['font.sans-serif'] = ['SimHei']


def plot_test(names, y_F1, y_P, y_R):
    # print("names = ", names)
    # names = ['5', '10', '15', '20', '25']
    x = range(len(names))
    # y = [0.855, 0.84, 0.835, 0.815, 0.81]
    # y1 = [0.86, 0.85, 0.853, 0.849, 0.83]
    # plt.plot(x, y, 'ro-')
    # plt.plot(x, y1, 'bo-')
    # pl.xlim(-1, 11)  # 限定横轴的范围
    # pl.ylim(-1, 110)  # 限定纵轴的范围
    # marker是折现交点的符号形式，label是函数说明，mec是交点符号边缘颜色，mfc是交点符号face颜色，ms是交点符号大小
    # plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'y=x^2曲线图')
    plt.plot(x, y_F1, marker='o', color='k', mec='k', ms=7, mfc='k', label=u'F1')
    plt.plot(x, y_P, marker='*', color='r', mec='r', ms=7, mfc='r', label=u'P')
    plt.plot(x, y_R, marker='v', color='y', mec='y', ms=7, mfc='y', label=u'R')
    # plt.plot(x, y_ACC, marker='+', color='b', mec='b', ms=7, mfc='b', label=u'ACC')
    # plt.xlim(xmin=0)
    plt.xlim(xmin=-0.9, xmax=7.8)
    plt.ylim(ymin=0.1, ymax=0.956)
    plt.legend(fontsize=17.5, loc=1)  # 让图例生效
    plt.xticks(x, names, rotation=0)
    plt.tick_params(labelsize=16)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.3f'))
    # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    # plt.xlabel(u"time(s)邻居")  # X轴标签
    # plt.ylabel("RMSE")  # Y轴标签
    # plt.title("A simple plot")  # 标题

    plt.show()


def read_csv(experiment_id):
    # data = pd.read_csv("C:\desktop\Research\选题相关\基于模糊系统和深度学习的在线评论情感计算\作图\\result\\result.csv")
    data = pd.read_csv("C:\desktop\Research\选题相关\基于模糊推理系统和深度学习的在线评论情感计算\作图\\result\\new_result.csv")
    data = data.loc[data["experiment_id"] == experiment_id]
    # data['aspect'] = data['aspect'].astype(int)
    names = list(data['aspect'])
    print("names = ", names)
    y_F1 = list(data['macro_f1_score'])
    y_P = list(data['macro_precision'])
    y_R = list(data['macro_recall'])
    # y_ACC = list(data['acc_score'])

    return names, y_F1, y_P, y_R


def read_csv2(experiment_id):
    # path = "C:\desktop\Research\选题相关\基于模糊系统和深度学习的在线评论情感计算\作图\\result\\new_result.csv"
    path = "C:\desktop\Research\选题相关\基于模糊推理系统和深度学习的在线评论情感计算\作图\result\\new_result.csv"
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        for row in reader:
            print("row = ", row)


def concatenate_vectors():
    vec1 = [[5, 2, 7, 8, 9], [9, 0, 6, 4, 3]]
    vec2 = [[1, 5, 2], [4, 2, 0]]

    vec3 = [[[5, 2, 7, 8, 9], [9, 0, 6, 4, 3]], [[51, 21, 71, 81, 91], [91, 1, 61, 41, 31]]]
    vec4 = [[[1, 5, 2], [4, 2, 0]], [[12, 52, 22], [42, 22, 20]]]

    print("vec1's type = ", type(vec1))
    print("vec2's type = ", type(vec2))

    con = numpy.concatenate((vec1, vec2), axis=1)

    print(con)


def my_padding():
    df = pd.DataFrame({"A": ["趁着国庆节。\n"
                             "一家人在白天在山里玩耍之后。\n"
                             "晚上决定吃李记搅团。", "东门外这家店门口停车太难了，\n"
                                           "根本没空位置，\n"
                                           "所以停在了旁边的地下停车场。", "还有一个汤", "还有一个汤2"], "B": [5,6,7,8], "C": [1,1,1,1]})
    print("df = ", df)
    print("df.loc[:2] = ", df.loc[:2])

    df2 = pd.DataFrame({"A": [5, 6, 7, 8], "B": [5, 6, 7, 8], "C": [1, 1, 1, 1], "D": [1, 2, 1, 4]})
    print("df2.loc[:2] = ", df2.loc[:2])

    ls = [1, 2, 3, 4, 5, 6, 7]
    print(ls[:2])

    df_array = np.array(df)
    print("df_array's type = ", type(df_array))
    print("df_array = ", df_array)


def deleteEnter():
    df = pd.DataFrame({"A": ["趁着国庆节。\n"
                             "一家人在白天在山里玩耍之后。\n"
                             "晚上决定吃李记搅团。", "东门外这家店门口停车太难了，\n"
                                           "根本没空位置，\n"
                                           "所以停在了旁边的地下停车场。\n", "还有一个汤", "还有一个汤2\n"], "B": [5, 6, 7, 8], "C": [1,1,1,1]})
    df2 = pd.DataFrame({"A": ["趁着国庆节。"
                              "一家人在白天在山里玩耍之后。"
                              "晚上决定吃李记搅团。", "东门外这家店门口停车太难了，"
                                            "根本没空位置，"
                                            "所以停在了旁边的地下停车场。", "还有一个汤", "还有一个汤2"], "B": [5, 6, 7, 8], "C": [1,1,1,1]})
    # print("df['A']_1 = ", df["A"])
    # print("df2 = ", df2)
    # df["A"] = df["A"].apply(lambda x: x.strip())
    # print("df['A']_2 = ", df["A"])
    # df["A"] = df["A"].apply(lambda x: x.replace("\n", "").replace('\r', ''))
    # print("df['A']_3 = ", df["A"])
    print(df)

    # 遍历清洗
    rows = len(df)
    print("rows = ", rows)
    for i in range(rows):
        current = df.loc[i, 'A']
        print("current_0 = ", current)
        current = current.replace('\n', '')
        print("current_1 = ", current)
        df.loc[i, 'A'] = current
    print("df_final = ", df)


def delete_enter(df):
    if "\n" in df["A"]:
        return df["A"]


def padding(text, maxlen):
    length = len(text)
    if length < 512:
        pass

# 将文本截取至maxlen-2的长度
def textsCut(input_texts, maxlen):
    result = []
    for text in input_texts:
        print("text = ", text)
        length = len(text)
        print("length before = ", length)
        if length <= maxlen - 2:
            result.append(text)
            print("length after1 = ", len(text))
            continue
        print("length after2 = ", len(text[:maxlen - 2]))
        result.append(text[:maxlen - 2])

    return result


def CutTest():
    texts = ["我是中国人", "你是中国人吗", "真的挺好玩的是不是", "挺好的", "今天吃的什么饭呀？好不好吃呀？"]
    texts = textsCut(texts, 8)
    print("*" * 100)
    for text in texts:
        print("text = ", text)
        print("text's length = ", len(text))


def train_test_split_test():
    X = ["我是中国人", "你是中国人吗", "真的挺好玩的是不是", "挺好的", "今天吃的什么饭呀？好不好吃呀？", "不开心", "不高兴", "今天吃的什么饭呀？好不好吃呀？", "不开心", "不高兴"]
    y = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
    X = np.array(X)
    y = np.array(y)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3)
    print("X_train = ", X_train)
    print("y_train = ", y_train)
    print("X_validation = ", X_validation)
    print("y_validation = ", y_validation)


if __name__ == "__main__":
    print(">>>in the main function of test.py...")

    deleteEnter()

    # name = "learning_rate"
    # read_csv(name)
    # names, y_F1, y_P, y_R = read_csv(name)
    # plot_test(names, y_F1, y_P, y_R)

    # adaption_predict("experiment_28.h5")
    # variable = "window"
    # names, y_F1, y_P, y_R = read_csv(variable)
    # print(names, y_F1, y_P, y_R)
    # plot_test(names, y_F1, y_P, y_R, )

    # data = dp_s.initData3()
    # print(data)

    # for i in range(10):
    #     print("hello world")
    # columns = ['id', 'review', 'service', 'price', 'self', 'outside', 'safety', 'inside', 'useful', 'times']
    # print(columns[2: -1])

    print(">>>in the end...")


import pandas as pd
import numpy as np
import csv
import codecs
import jieba
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support as score, classification_report
from sklearn.utils import shuffle

# 之后将所有数据放入一个文件
data_path = "datasets/farmer/training_corn_apple.csv"
# data_path = "datasets/farmer/training_corn_apple.csv"
data_path_debug = "datasets/farmer/training_debug.csv"

debug_data_length = 100

# 结果保存路径
result_path = "result/farmer/complaints/sentence_sentiment_corn_apple.csv"
result_path_debug = "result/farmer/complaints/sentence_sentiment_farmer_debug.csv"

stoplist = pd.read_csv('config/stopwords.txt').values

# 词典长度
dict_length = 150000  # 词典的长度

# 腾讯词向量
pre_word_embedding = "config/preEmbeddings/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"


# 从文件中读取数据
def readFromCSV(debug=False):
    data = pd.read_csv(data_path)
    data = shuffle(data)

    if debug:
        data = data[:debug_data_length]

    X = np.array(data["reviews"])
    Y = np.array(data["label"])
    # print("X's type = ", type(X))
    # print("Y's type = ", type(Y))

    return data, X, Y


# 从文件中读取数据，保存tag
def readFromCSV2(debug=False):
    if debug:
        path = data_path_debug
    else:
        path = data_path
    data = pd.read_csv(path)
    data = shuffle(data)

    return data


# 使用腾讯词向量表示X
def sentence2vector(X, debug):
    if debug:
        return [[0] * 300] * len(X)

    f = open(pre_word_embedding, "r", encoding="utf-8")
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

    X_result = []
    for line in X:
        if len(line) == 0:
            print("当前评论文本为空...")
        current_vector = []
        for word in line:
            temp = embeddings_index.get(word)
            if temp is not None:
                current_vector.append(temp)
        if len(current_vector) == 0:
            current_vector = [[0] * 300]
        X_result.append(np.array(current_vector).mean(axis=0))

    return np.array(X_result)


# 根据预测结果计算各种指标
def calculateScore(Y_validation, Y_predicts, model_name, debug=False):
    # print("Y_validation = ", Y_validation)
    # print("Y_predicts = ", Y_predicts)
    report = classification_report(Y_validation, Y_predicts, digits=4, output_dict=True)
    print(report)

    # 保存至文件
    print("》》》正在将结果保存至文件。。。")
    save_result_to_csv(report, model_name, debug)


# 结果保存至文件
def save_result_to_csv(report, model_name, debug=False):
    accuracy = report.get("accuracy")

    macro_avg = report.get("macro avg")
    macro_precision = macro_avg.get("precision")
    macro_recall = macro_avg.get("recall")
    macro_f1 = macro_avg.get('f1-score')

    weighted_avg = report.get("weighted avg")
    weighted_precision = weighted_avg.get("precision")
    weighted_recall = weighted_avg.get("recall")
    weighted_f1 = weighted_avg.get('f1-score')
    data = [model_name, weighted_precision, weighted_recall, weighted_f1, macro_precision, macro_recall, macro_f1, accuracy]

    if debug:
        path = result_path_debug
    else:
        path = result_path
    with codecs.open(path, "a", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)
        f.close()


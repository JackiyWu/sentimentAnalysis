#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba
import re
import csv
from collections import Counter
import codecs
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
import keras
from sklearn.utils import shuffle
import KMeansCluster as KMC
import fuzzySystem as fsys
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report


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


# 读取数据
def initData():
    print("in initData function of dataProcess_sentence.py...")
    data = pd.read_csv("data2/ChnSentiCorp_htl_all.csv")
    # data = data[:10]
    # print("data shuffle before = ", data)

    # 打乱顺序
    data = shuffle(data)
    # print("data shuffle after = ", data)

    # data = data[6000:]

    # y_cols = data.columns.values.tolist()[2:22]
    y_cols = data.columns.values.tolist()
    print(y_cols)
    print("end of initData function in dataProcess.py...")
    # print("data'type = ", type(data))

    return data, y_cols


# 处理百度点石数据集，1-物流快递，2-医疗服务，3-旅游住宿，4-金融服务，5-食品餐饮
def initData2(type):
    columns = ['id', 'type', 'review', 'label']
    data = pd.read_csv("C:\desktop\Research\DataSet\百度点石大赛\data_train.csv", sep='\t', names=columns, encoding='utf-8')
    if type == 1:
        data = data.loc[data['type'] == str('物流快递')]
    elif type == 2:
        data = data.loc[data['type'] == str('医疗服务')]
    elif type == 3:
        data = data.loc[data['type'] == str('旅游住宿')]
    elif type == 4:
        data = data.loc[data['type'] == str('金融服务')]
    else:
        data = data.loc[data['type'] == str('食品餐饮')]

    y_cols = data.columns.values.tolist()

    data = shuffle(data)

    # data = data[:10]

    return data, y_cols


# 接收1个参数，debug
# 生成所有领域数据构成的数据集
# field id:物流-0，餐饮-1，医疗-2，金融-3，旅游-4
def initDataForFive(sampling, debug=False):
    print(">>>in the function of initDataForFive...")

    columns = ['id', 'type', 'review', 'label']
    data = pd.read_csv("datasets/baidu/data_train.csv", sep='\t', names=columns, encoding='utf-8')
    print("initData4 data's length = ", len(data))

    data_logistics = data.loc[data['type'] == str("物流快递")]
    print("data_logistics's length = ", len(data_logistics))
    data_catering = data.loc[data['type'] == str("食品餐饮")]
    data_medical = data.loc[data['type'] == str("医疗服务")]
    data_financial = data.loc[data['type'] == str("金融服务")]
    data_traveling = data.loc[data['type'] == str("旅游住宿")]

    y_cols = data.columns.values.tolist()
    all_data = pd.concat([data_logistics, data_catering, data_medical, data_financial, data_traveling])
    data = pd.concat([data_logistics, data_catering, data_medical, data_financial, data_traveling])
    '''
    all_data = shuffle(all_data)
    # 是否下采样
    if sampling:
        ratio = 0.5
        all_data = lower_sampling(all_data, ratio)

    data = shuffle(data)
    '''
    field = "allFields"

    return data, y_cols, all_data, field


# 接收四个参数
def initDataForThree(id1, id2, id3, sampling):
    print(">>>in the function of initDataForThree...")
    columns = ['id', 'type', 'review', 'label']
    data = pd.read_csv("datasets/baidu/data_train.csv", sep='\t', names=columns, encoding='utf-8')
    print("initData4 data's length = ", len(data))

    data_logistics = data.loc[data['type'] == str("物流快递")]
    print("data_logistics's length = ", len(data_logistics))
    data_catering = data.loc[data['type'] == str("食品餐饮")]
    data_medical = data.loc[data['type'] == str("医疗服务")]
    data_financial = data.loc[data['type'] == str("金融服务")]
    data_traveling = data.loc[data['type'] == str("旅游住宿")]

    y_cols = data.columns.values.tolist()

    dict_names = {0: 'logistics', 1: 'catering', 2: 'medical', 3: 'financial', 4: 'traveling'}
    dict_datas = {0: data_logistics, 1: data_catering, 2: data_medical, 3: data_financial, 4: data_traveling}
    data_1 = dict_datas.get(id1)
    data_2 = dict_datas.get(id2)
    data_3 = dict_datas.get(id3)
    name_1 = dict_names.get(id1)
    name_2 = dict_names.get(id2)
    name_3 = dict_names.get(id3)
    all_data = pd.concat([data_1, data_2, data_3])

    # 是否下采样
    if sampling:
        ratio = 0.5
        all_data = lower_sampling(all_data, ratio)

    all_data = shuffle(all_data)
    field = name_1 + name_2 + name_3
    data = all_data

    return data, y_cols, all_data, field


# 接收三个参数，debug和ids
# field id:物流-0，餐饮-1，医疗-2，金融-3，旅游-4
def initDataForTwo(id1, id2, sampling, debug=False):
    print(">>>in the function of initDataForTwo...")
    columns = ['id', 'type', 'review', 'label']
    data = pd.read_csv("datasets/baidu/data_train.csv", sep='\t', names=columns, encoding='utf-8')
    print("initData4 data's length = ", len(data))

    data_logistics = data.loc[data['type'] == str("物流快递")]
    print("data_logistics's length = ", len(data_logistics))
    data_catering = data.loc[data['type'] == str("食品餐饮")]
    data_medical = data.loc[data['type'] == str("医疗服务")]
    data_financial = data.loc[data['type'] == str("金融服务")]
    data_traveling = data.loc[data['type'] == str("旅游住宿")]

    y_cols = data.columns.values.tolist()

    dict_names = {0: 'logistics', 1: 'catering', 2: 'medical', 3: 'financial', 4: 'traveling'}
    dict_datas = {0: data_logistics, 1: data_catering, 2: data_medical, 3: data_financial, 4: data_traveling}
    data_1 = dict_datas.get(id1)
    data_2 = dict_datas.get(id2)
    name_1 = dict_names.get(id1)
    name_2 = dict_names.get(id2)
    all_data = pd.concat([data_1, data_2])

    # 是否下采样
    if sampling:
        ratio = 0.5
        all_data = lower_sampling(all_data, ratio)

    all_data = shuffle(all_data)
    field = name_1 + name_2
    data = all_data

    return data, y_cols, all_data, field


# 接收两个参数，debug和排除的领域数据
# field id:物流-0，餐饮-1，医疗-2，金融-3，旅游-4
def initDataForFour(field_id, sampling, debug=False):
    print(">>>in the function of initDataForFour...")
    columns = ['id', 'type', 'review', 'label']
    data = pd.read_csv("datasets/baidu/data_train.csv", sep='\t', names=columns, encoding='utf-8')
    print("initData4 data's length = ", len(data))

    data_logistics = data.loc[data['type'] == str("物流快递")]
    print("data_logistics's length = ", len(data_logistics))
    data_catering = data.loc[data['type'] == str("食品餐饮")]
    data_medical = data.loc[data['type'] == str("医疗服务")]
    data_financial = data.loc[data['type'] == str("金融服务")]
    data_traveling = data.loc[data['type'] == str("旅游住宿")]

    y_cols = data.columns.values.tolist()

    dict_names = {0: 'logistics', 1: 'catering', 2: 'medical', 3: 'financial', 4: 'traveling'}
    dict_datas = {0: data_logistics, 1: data_catering, 2: data_medical, 3: data_financial, 4: data_traveling}
    field = ""
    temp = []
    for i in range(5):
        if i == field_id:
            continue
        field += dict_names.get(i)
        temp.append(dict_datas.get(i))
    all_data = pd.concat(temp)

    # 下采样
    if sampling:
        ratio = 0.5
        all_data = lower_sampling(all_data, ratio)

    all_data = shuffle(all_data)
    data = all_data

    return data, y_cols, all_data, field


# 接收两个参数，debug和数据集id（单个数据集）
# field id:物流-0，餐饮-1，医疗-2，金融-3，旅游-4
def initDataForOne(field_id, debug=False):
    print(">>>in the function of initDataForOne...")

    columns = ['id', 'type', 'review', 'label']
    data = pd.read_csv("datasets/baidu/data_train.csv", sep='\t', names=columns, encoding='utf-8')
    print("initData4 data's length = ", len(data))

    data_logistics = data.loc[data['type'] == str("物流快递")]
    print("data_logistics's length = ", len(data_logistics))
    data_catering = data.loc[data['type'] == str("食品餐饮")]
    data_medical = data.loc[data['type'] == str("医疗服务")]
    data_financial = data.loc[data['type'] == str("金融服务")]
    data_traveling = data.loc[data['type'] == str("旅游住宿")]

    y_cols = data.columns.values.tolist()
    if field_id == 0:
        field = "logistics"
        data = data_logistics
        all_data = pd.concat([data_logistics, data_catering, data_medical, data_financial, data_traveling])
    elif field_id == 1:
        field = "catering"
        data = data_catering
        all_data = pd.concat([data_catering, data_logistics, data_medical, data_financial, data_traveling])
    elif field_id == 2:
        field = "medical"
        data = data_medical
        all_data = pd.concat([data_medical, data_logistics, data_catering, data_financial, data_traveling])
    elif field_id == 3:
        field = "financial"
        data = data_financial
        all_data = pd.concat([data_financial, data_logistics, data_catering, data_medical, data_traveling])
    elif field_id == 4:
        field = "traveling"
        data = data_traveling
        all_data = pd.concat([data_traveling, data_logistics, data_catering, data_medical, data_financial])

    data = shuffle(data)

    return data, y_cols, all_data, field


# 接收两个入参，将第二个领域的中性语料放入第一个里面，升采样
def initData3(debug=False):
    columns = ['id', 'type', 'review', 'label']
    data = pd.read_csv("datasets/baidu/data_train.csv", sep='\t', names=columns, encoding='utf-8')
    print("initData3 data's length = ", len(data))

    # 将data2的中性语料加入到data1中,0-负向，1-中性，2-正向

    # data1 = data.loc[data['type'] == str("医疗服务")]
    data_logistics = data.loc[data['type'] == str("物流快递")]
    data_catering = data.loc[data['type'] == str("食品餐饮")]
    print("data_logistics's length = ", len(data_logistics))
    data_medical = data.loc[data['type'] == str("医疗服务")]
    data_financial = data.loc[data['type'] == str("金融服务")]
    data_traveling = data.loc[data['type'] == str("旅游住宿")]
    data2 = data.loc[data['type'] != str("物流快递")]
    data2 = data2.loc[data['label'] == 1]
    print("data2's length = ", len(data2))
    # print("data1 = ", data1)
    # print("data2 = ", data2)

    ratio = 0.5
    # data_medical = lower_sampling(data_medical, ratio)
    # data_financial = lower_sampling(data_financial, ratio)
    # data_traveling = lower_sampling(data_traveling, ratio)
    # data2 = lower_sampling(data2, ratio)

    # data = pd.concat([data_logistics, data2])
    data = pd.concat([data_logistics, data_catering, data_traveling, data_financial])
    # data = lower_sampling(data, ratio)
    data = shuffle(data)

    print("data_catering's length = ", len(data))

    data_medical = shuffle(data_medical)
    data_financial = shuffle(data_financial)
    data_traveling = shuffle(data_traveling)
    print("data_medical's length = ", len(data_medical))
    print("data_financial's length = ", len(data_financial))
    print("data_traveling's length = ", len(data_traveling))

    # all_data数据是总体的，没有打乱的
    all_data = pd.concat([data, data_financial, data_traveling, data_medical])

    # if debug:
    #     data = data[:50]

    y_cols = data.columns.values.tolist()

    # return data, y_cols
    return data, y_cols, data_medical, data_financial, data_traveling, all_data


# ratio为保留的比例
def lower_sampling(data, ratio):
    print(">>>in lower_sampling function...")
    neutral_data = data[data['label'] == 1]
    negative_data = data[data['label'] == 0]
    positive_data = data[data['label'] == 2]

    neutral_length = len(neutral_data)
    negative_length = len(negative_data)
    positive_length = len(positive_data)
    min_length = min(neutral_length, negative_length, positive_length)
    print("min_length = ", min_length)
    print("positive_length = ", positive_length)

    if (neutral_length * ratio) > min_length:
        index = np.random.randint(len(neutral_data), size=int(min(neutral_length * ratio, min_length / ratio)))
        print("index's length = ", len(index))
        print("neutral_data.shape = ", neutral_data.shape)
        neutral_data = neutral_data.iloc[list(index)]
        print("neutral_data.shape = ", neutral_data.shape)
    if (negative_length * ratio) > min_length:
        index = np.random.randint(len(negative_data), size=int(min(negative_length * ratio, min_length / ratio)))
        print("negative_data.shape = ", negative_data.shape)
        print("index's length = ", len(index))
        negative_data = negative_data.iloc[list(index)]
        print("negative_data.shape = ", negative_data.shape)
    if (positive_length * ratio) > min_length:
        index = np.random.randint(len(positive_data), size=int(min(positive_length * ratio, min_length / ratio)))
        print("index's length = ", len(index))
        print("positive_data.shape = ", positive_data.shape)
        positive_data = positive_data.iloc[list(index)]
        print("positive_data.shape = ", positive_data.shape)

    final_data = pd.concat([neutral_data, negative_data, positive_data])
    print("final_data.shape = ", final_data.shape)

    print(">>>end of lower_sampling function...")
    return final_data


# 输入细粒度的源数据，生成包括6个粗粒度属性的数据，存入csv
# origin_data的2-21列是分细粒度属性的标签列。2-4：位置location，5-8：服务service，9-11：价格price，12-15：环境environment，16-18：菜品dish，19-21：其他others
# 计算规则：①
def processDataToTarget(origin_data):
    print("in processDataToTarget function of dataProcess.py...")
    # header = ["id", "content", "location", "service", "price", "environment", "dish"]
    header = ["id", "content", "location", "service", "price", "environment", "dish", "others"]

    data = []

    for index, row in origin_data.iterrows():
        d = [row["id"], row["content"]]

        location = [row["location_traffic_convenience"], row["location_distance_from_business_district"], row["location_easy_to_find"]]
        service = [row["service_wait_time"], row["service_waiters_attitude"], row["service_parking_convenience"], row["service_serving_speed"]]
        price = [row["price_level"], row["price_cost_effective"], row["price_discount"]]
        environment = [row["environment_decoration"], row["environment_noise"], row["environment_space"], row["environment_cleaness"]]
        dish = [row["dish_portion"], row["dish_taste"], row["dish_look"], row["dish_recommendation"]]
        others = [row["others_overall_experience"], row["others_willing_to_consume_again"]]

        location = processLabel(location)
        service = processLabel(service)
        price = processLabel(price)
        environment = processLabel(environment)
        dish = processLabel(dish)
        others = processLabel(others)

        d.append(location)
        d.append(service)
        d.append(price)
        d.append(environment)
        d.append(dish)
        d.append(others)

        data.append(d)
        # print("d = ", d)
    print("data = ", data)
    # with codecs.open("data/sentiment_analysis_validation_set_new_without_others.csv", "wb", "utf-8") as f:
    with codecs.open("data/sentiment_analysis_training_set_new_without_others.csv", "w", "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
        f.close()

    print("end of processDataToTarget function in dataProcess.py...")


# 传入细粒度属性的label，输出粗粒度属性label
def processLabel(aspect):
    # print("aspect = ", aspect)
    if -2 not in aspect:  # 如果标签没有-2，则求整体平均
        average = np.average(np.array(aspect))
    else:  # label中有-2分为两种情况
        length = len(set(aspect))

        if length > 1:  # 不全是-2
            while -2 in aspect:
                aspect.remove(-2)
            average = np.average(aspect)
        else:  # 全是-2
            average = -2

    # 如果average非整数&≠-2
    if average != -2:
        if average > 0:
            average = 1
        elif average < 0:
            average = -1
        else:
            average = 0

    # print("average = ", average)
    return average


# 根据类别label数量，计算得到class_weight
def calculate_class_weight(labels):
    print(">>>in calculate_class_weights function of dataProcess.py...")
    print("labels = ", labels)
    class_weights = []
    for label in labels:
        max_value = max(label.values())
        print(label.get(-2), label.get(-1), label.get(0), label.get(1))
        class_weights.append({0: max_value / label.get(-2), 1: max_value / label.get(-1), 2: max_value / label.get(0), 3: max_value / label.get(1)})

    print("class_weights = ", class_weights)

    print(">>>end of calculate_class_weights function in dataProcess.py...")

    return class_weights


def getStopList():
    stoplist = pd.read_csv('stopwords.txt').values
    return stoplist


# 处理数据得到输入语料的文本
def processDataToTexts(data, stoplist):
    print(">>>in processDataToTexts of dataProcess.py...")
    # print("data[review] = ", data["review"])

    # 去标点符号
    print(">>>去标点符号ing。。。")
    data['words'] = data['review'].apply(lambda x: re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？、~@#￥%……&*（）．；：】【|]+", "", str(x)))
    # jieba分词
    print(">>>jieba分词ing。。。")
    data['words'] = data['words'].apply(lambda x: list(jieba.cut(x)))

    # 去掉开头
    data['words'] = data['words'].apply(lambda x: x[1:-1])
    data['len'] = data['words'].apply(lambda x: len(x))
    maxlen = data['len'].max()
    words_dict = []
    texts = []

    # 去掉停用词
    print(">>>去停用词ing in DataProcess.py...")
    for index, row in data.iterrows():
        line = [word.strip() for word in list(row['words']) if word not in stoplist]

        words_dict.extend([word for word in line])
        texts.append(line)

    # print("texts = ", texts)
    # print("data[words] = ", data["words"])

    print(">>>end of processDataToTexts in dataProcess.py...")

    return texts


# 处理数据生成训练集 验证集 测试集
# def processData(data, stoplist, dict_length, maxlen, ratios):
def processData(data, stoplist, dict_length, maxlen, ratios, data_medical, data_financial, data_traveling):
    print(">>>in processData function...")

    # print(words_dict)
    # print(texts)
    texts = processDataToTexts(data, stoplist)
    texts_medical = processDataToTexts(data_medical, stoplist)
    texts_financial = processDataToTexts(data_financial, stoplist)
    texts_traveling = processDataToTexts(data_traveling, stoplist)
    '''
    '''

    stop_data = pd.DataFrame(texts)

    # 利用keras的Tokenizer进行onehot，并调整未等长数组
    tokenizer = Tokenizer(num_words=dict_length)
    tokenizer.fit_on_texts(texts)

    word_index = tokenizer.word_index
    # print("word_index = ", word_index)

    data_w = tokenizer.texts_to_sequences(texts)
    data_T = sequence.pad_sequences(data_w, maxlen=maxlen)
    data_w_medical = tokenizer.texts_to_sequences(texts_medical)
    data_T_medical = sequence.pad_sequences(data_w_medical, maxlen=maxlen)
    data_w_financial = tokenizer.texts_to_sequences(texts_financial)
    data_T_financial = sequence.pad_sequences(data_w_financial, maxlen=maxlen)
    data_w_traveling = tokenizer.texts_to_sequences(texts_traveling)
    data_T_traveling = sequence.pad_sequences(data_w_traveling, maxlen=maxlen)
    '''
    '''

    # 数据划分，重新划分为训练集，测试集和验证集
    data_length = data_T.shape[0]
    print("data_length = ", data_length)
    size_train = int(data_length * ratios[0])
    size_val = int(data_length * ratios[1])

    global dealed_train
    global dealed_val
    global dealed_test
    dealed_train = data_T[: size_train]
    dealed_val = data_T[size_train: (size_train + size_val)]
    dealed_test = data_T[(size_train + size_val):]

    dealed_val_medical = data_T_medical
    dealed_val_financial = data_T_financial
    dealed_val_traveling = data_T_traveling
    '''
    '''

    global train  # 训练数据集，包括语料、标签、id、len（后期增加的）
    global val  # 验证数据集，包括语料、标签、id、len（后期增加的）
    global test  # 测试数据集，包括语料、标签、id、len（后期增加的）
    train = data[: size_train]
    val = data[size_train: (size_train + size_val)]
    test = data[(size_train + size_val):]
    # print(train.shape)
    # print(val.shape)
    # print(test.shape)
    # print("test = ", test)
    val_medical = data_medical
    val_financial = data_financial
    val_traveling = data_traveling
    '''
    '''

    print(">>>end of processData function...")

    # return dealed_train, dealed_val, dealed_test, train, val, test, texts, word_index
    return dealed_train, dealed_val, dealed_test, train, val, test, texts, word_index, dealed_val_medical, dealed_val_financial, dealed_val_traveling, val_medical, val_financial, val_traveling


# 处理数据生成训练集 验证集
def processData3(origin_data, stoplist, dict_length, maxlen, ratios):
    print(">>>in processData3 function...")
    print("all_data's length = ", len(origin_data))

    all_texts = processDataToTexts(origin_data, stoplist)
    stop_data = pd.DataFrame(all_texts)

    # 利用keras的Tokenizer进行onehot，并调整未等长数组
    tokenizer = Tokenizer(num_words=dict_length)
    tokenizer.fit_on_texts(all_texts)

    word_index = tokenizer.word_index

    data_w = tokenizer.texts_to_sequences(all_texts)
    data_T = sequence.pad_sequences(data_w, maxlen=maxlen)

    # 数据划分，重新划分为训练集，测试集和验证集
    data_length = data_T.shape[0]
    print("data_length = ", data_length)

    size_train = int(data_length * ratios[0])
    print("size_train's length = ", size_train)

    # 数据划分，重新划分为训练集，测试集和验证集
    global dealed_train
    global dealed_val
    dealed_train = data_T[: size_train]
    dealed_val = data_T[size_train:]

    train = origin_data[: size_train]
    val = origin_data[size_train:]
    print("val's length = ", len(val))

    print(">>>end of processData function...")

    return dealed_train, dealed_val, train, val, word_index, data_T


# 处理数据生成训练集 验证集 测试集
# 处理输入预料，生成训练集、验证集、测试集，其中训练集即为餐饮业+物流业数据，验证集分别为医疗业、金融业、旅游业数据
def processData2(all_data, stoplist, dict_length, maxlen, ratios, dealed_train_fuzzy_concat, catering_length, medical_length, financial_length, traveling_length):
    print(">>>in processData2 function...")

    # all_data = np.concatenate((data, dealed_train_fuzzy_concat), axis=0)
    print("all_data's length = ", len(all_data))

    all_texts = processDataToTexts(all_data, stoplist)
    stop_data = pd.DataFrame(all_texts)

    # 利用keras的Tokenizer进行onehot，并调整未等长数组
    tokenizer = Tokenizer(num_words=dict_length)
    tokenizer.fit_on_texts(all_texts)

    # print(words_dict)
    # print(texts)

    word_index = tokenizer.word_index
    # print("word_index = ", word_index)

    data_w = tokenizer.texts_to_sequences(all_texts)
    data_T = sequence.pad_sequences(data_w, maxlen=maxlen)

    # 数据划分，重新划分为训练集，测试集和验证集
    data_length = data_T.shape[0]
    print("data_length = ", data_length)
    # if data_length != dealed_train_fuzzy_length:
    #     print(">>>>>>>>>>出错啦！！！训练集长度有问题！！！！")
    size_train = int(catering_length * ratios[0])
    print("size_train's length = ", size_train)
    print("catering_length = ", catering_length)

    # 数据划分，重新划分为训练集，测试集和验证集
    global dealed_train
    global dealed_val
    dealed_train = data_T[: size_train]
    dealed_val = data_T[size_train: catering_length]

    dealed_val_1 = data_T[catering_length: medical_length + catering_length]
    dealed_val_2 = data_T[medical_length + catering_length: medical_length + catering_length + financial_length]
    dealed_val_3 = data_T[medical_length + catering_length + financial_length:]
    print("dealed_val_1's length = ", len(dealed_val_1))
    print("dealed_val_2's length = ", len(dealed_val_2))
    print("dealed_val_3's length = ", len(dealed_val_3))
    print("data_T's length = ", len(data_T))
    print("medical_length + catering_length + financial_length's length = ", medical_length + catering_length + financial_length)

    train = all_data[: size_train]
    val = all_data[size_train: catering_length]
    print("val's length = ", len(val))
    val_1 = all_data[catering_length: medical_length + catering_length]
    val_2 = all_data[medical_length + catering_length: medical_length + catering_length + financial_length]
    val_3 = all_data[medical_length + catering_length + financial_length:]
    print("val_1's length = ", len(val_1))
    print("val_2's length = ", len(val_2))
    print("val_3's length = ", len(val_3))

    print(">>>end of processData function...")

    return dealed_train, dealed_val, dealed_val_1, dealed_val_2, dealed_val_3, train, val, val_1, val_2, val_3, word_index


# 确定maxlen
def calculate_maxlen(texts):
    maxlen = 0

    # 直接取最长的评论长度
    '''
    for line in texts:
        if maxlen < len(line):
            maxlen = len(line)
    '''

    # 取评论长度的平均值+两个评论的标准差（假设评论长度的分布满足正态分布，则maxlen可以涵盖95左右的样本）
    lines_length = [len(line) for line in texts]
    lines_length = np.array(lines_length)

    maxlen = np.mean(lines_length) + 2 * np.std(lines_length)
    maxlen = int(maxlen)

    return maxlen


# 传入语料数据，输出不同属性-label的样本数,依次为位置、服务、价格、环境、菜品、其他
def calculate_sample_number(origin_data):
    aspects = ["location", "service", "price", "environment", "dish", "others"]

    result = []
    for aspect in aspects:
        current = list(origin_data[aspect])
        # print(aspect, " = ", current)
        result.append(Counter(current))

    return result


def build_model(max_words, embedding_dim, maxlen):
    print(">>>Building the model...")
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(5))
    # model.add(Dropout(0.5))
    model.add(Conv1D(64, 3, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    # model.add(layers.Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    return model


def build_model2(max_words, embedding_dim, maxlen):
    print(">>>Building the model...")
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    conv1 = Conv1D(64, 3, activation='relu')
    conv11 = Conv1D(64, 3, activation='relu')
    added = keras.layers.add([conv1, conv11])
    model.add(added)
    pooling1 = MaxPooling1D(5)
    model.add(pooling1)
    # model.add(Dropout(0.5))
    conv2 = Conv1D(64, 3, activation='relu')
    model.add(conv2)
    # model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    # model.add(layers.Dense(32, activation='relu'))
    dense = Dense(4, activation='softmax')
    model.add(dense)
    return model


def train_CNN(train_x, test_x, val_x, y_cols, debug=False, folds=1):
    print(">>>in train_CNN function...")
    # print(train_x, test_x, val_x, y_cols)
    # print("train:", train)
    # print("val:", val)
    # print("test:", test)
    model = build_model()

    print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    F1_scores = 0
    F1_score = 0
    result = {}
    if debug:
        y_cols = ['location_traffic_convenience']
    for index, col in enumerate(y_cols):
        train_y = train[col] + 2
        val_y = val[col] + 2
        y_val_pred = 0
        y_test_pred = 0
        #         epochs=[5,10]   , stratify=train_y
        for i in range(folds):
            y_train_onehot = to_categorical(train_y)
            history = model.fit(train_x, y_train_onehot, epochs=3, batch_size=64, validation_split=0.2)

            # 预测验证集和测试集
            y_val_pred = model.predict(val_x)
            y_test_pred += model.predict(test_x)

        y_val_pred = np.argmax(y_val_pred, axis=1)

        F1_score = f1_score(y_val_pred, val_y, average='macro')
        F1_scores += F1_score

        print('第', index, '个细粒度', col, 'f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, val_y))
        y_test_pred = np.argmax(y_test_pred, axis=1)
        result[col] = y_test_pred-2
    print('all F1_score:', F1_scores/len(y_cols))
    print("result:", result)
    return result


# 用训练的模型预测数据，入参是模型、语料、真实标签，输出准确率、召回率、F1
def adaption_predict(path):
    print(">>>in adaption_predict function of dataProcess_sentence.py...")
    model = load_model(path)

    print(model.summary())

    ratios = [0.98, 0.01]
    dict_length = 150000  # 词典的长度，后期可以测试？？
    origin_data, y_cols = initData2(1)
    stoplist = getStopList()
    word_feature = KMC.read_excel2()
    input_texts = processDataToTexts(origin_data, stoplist)
    # 首先需要根据输入语料获取其模糊特征及词向量表示
    input_word_feature = fsys.calculate_sentiment_words_feature(input_texts, word_feature)
    dealed_train_fuzzy, dealed_val_fuzzy, dealed_test_fuzzy = fsys.calculate_fuzzy_feature(input_word_feature, ratios)
    input_texts = processDataToTexts(origin_data, stoplist)
    fuzzy_maxlen = fsys.calculate_input_dimension(dealed_train_fuzzy)
    # maxlen要保证和模型训练的时候句子长度一致
    maxlen = 53
    print("maxlen = ", maxlen)

    dealed_train, dealed_val, dealed_test, train, val, test, texts, word_index = processData(origin_data, stoplist,
                                                                                                  dict_length, maxlen, ratios)

    print("dealed_train.shape = ", dealed_train.shape)
    print("dealed_train_fuzzy.shape = ", dealed_train_fuzzy.shape)
    print("train.shape = ", train.shape)

    labels = train["label"]

    y_val_pred = model.predict([dealed_train_fuzzy, dealed_train])

    y_val_pred = np.argmax(y_val_pred, axis=1)

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("val_y = ", val_y)
    print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(labels, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    report = classification_report(labels, y_val_pred, digits=4, output_dict=True)

    print(report)

    print(">>>end of adaption_predict function in dataProcess_sentence.py...")


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import absa_config as config
import codecs
from keras_bert import Tokenizer as bert_Tokenizer, load_trained_model_from_checkpoint, extract_embeddings
import os
import numpy as np
import keras
from keras.layers import Input, Dense, Lambda, Flatten, Conv1D, MaxPool1D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers.merge import concatenate
import tensorflow as tf
from keras.models import save_model, load_model

# 句子的最大长度
MAXLEN = 512


def get_texts():
    texts = {"标准间太差房间还不如3星的而且设施非常陈旧.建议酒店把老的标准间从新改善.",
                 "设施还将就,但服务是相当的不到位,休息了一个晚上我白天出去,中午回来的时候居然房间都没有整理,尽管我挂了要求整理房间的牌子.",
                 "房间的格局在风水中属败相,卫生间马桶堵塞,无奈的一晚!!!",
                 "出差入住的酒店,订了个三人间.房间没空调,冷得要死,而且被子很潮.火车站旁,步行可到.",
                 "不太好，晚上被卡拉OK的声音干扰到很晚。楼下的饭店也搞出很多污染。",
                 "网上评价不错，但是实际情况并不尽如人意，与2-3星的水平差不多~~总的来说，性价比不高！",
                 "总台服务还可以，房间实在不怎么样，空调很吵，床居然是一边高一边低的，这个房价实在是不值得，下次不会选择。",
                 "总台服务还可以，房间实在不怎么样，空调很吵，床居然是一边高一边低的，这个房价实在是不值得",
                 "我感觉不行。。。性价比很差。不知道是银川都这样还是怎么的！",
                 "这个如家实在是太小了,房间很小很小,就不舒服了",
                 "CBD中心,周围没什么店铺,说5星有点勉强.不知道为什么卫生间没有电吹风",
                 "总的来说，这样的酒店配这样的价格还算可以，希望他赶快装修，给我的客人留些好的印象",
                 "价格比比较不错的酒店。这次免费升级了，感谢前台服务员。房子还好，地毯是新的，比上次的好些。早餐的人很多要早去些。",
                 "不错，在同等档次酒店中应该是值得推荐的！",
                 "前台楼层服务员都不错，房间安静整洁，交通方便，吃的周围也挺多．唯一不足，卫生间地漏设计不好，导致少量积水．",
                 "距离川沙公路较近,但是公交指示不对,如果是蔡陆线的话,会非常麻烦.建议用别的路线.房间较为简单.",
                 "位置不错,在市中心.周围吃饭等很方便.房间一如既往的干净",
                 "商务大床房，房间很大，床有2M宽，整体感觉经济实惠不错!",
                 "房间还可以，楼下有食街，服务态度很好！不过网络不太好，总是断线；早餐一般般。",
                 "服务很热情，交通也很便利，下次如果去北京我还会选择这家酒店!"}
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    return texts, labels


def get_texts_2():
    texts = {"差！非常差！不会再去了！！！", "感觉还不错，客服态度挺好的，赞！", "一般般吧，还好价格不算贵", "中规中矩，也算对得起这个价格了"}

    return texts


def load_data(tokenizer, texts):

    # 使用tokenizer对文本进行编码，然后转为numpy数组输出
    texts_list_1 = []
    texts_list_2 = []
    for text in texts:
        # tokens = tokenizer.tokenize(text)
        indices, segments = tokenizer.encode(first=text, max_len=512)
        texts_list_1.append(np.array(indices))
        # print("indices' length = ", len(indices))
        texts_list_2.append(np.array(segments))
        # print("segments' length = ", len(segments))

    print(">>>数据处理完毕。。。")
    print("texts_list_1's length = ", len(texts_list_1))
    print("texts_list_1[0]'s length = ", len(texts_list_1[0]))
    print("texts_list_2's length = ", len(texts_list_2))
    print("texts_list_2[0]'s length = ", len(texts_list_2[0]))

    return np.array(texts_list_1), np.array(texts_list_2)


def create_concat_model():
    print(">>>开始构造融合模型...")
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path, trainable=False)

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x_bert = bert_model([x1_in, x2_in])
    x_bert = Lambda(lambda x: x[:, 0], name='last_layer_1')(x_bert)  # 取出[CLS]对应的向量用来做分类

    x3_in = Input(shape=(3,))
    x_fuzzy = Dense(3, activation='linear', name='dense1_fuzzy')(x3_in)

    x = concatenate([x_bert, x_fuzzy], axis=-1)

    p = Dense(2, activation='softmax')(x)

    model = Model([x1_in, x2_in, x3_in], p)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    print(">>>融合模型加载结束。。。")
    model.summary()

    return model


# 三维融合模型
def create_concat_model2():
    print(">>>开始构造融合模型2...")
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path, trainable=False)

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x_bert = bert_model([x1_in, x2_in])
    # x_bert = Lambda(lambda x: x[:, 0], name='last_layer_1')(x_bert)  # 取出[CLS]对应的向量用来做分类

    # x3_in是每个字符隶属于三个聚类中心（负向、中性、正向）的程度
    x3_in = Input(shape=(512, 3,))
    # x_fuzzy = Dense(3, activation='linear', name='dense1_fuzzy')(x3_in)

    x = concatenate([x_bert, x3_in], axis=-1)

    cnn = Conv1D(32, 6, name='conv')(x)
    cnn = MaxPool1D(name='max_pool')(cnn)

    flatten = Flatten()(cnn)

    x = Dense(32, activation='relu', name='dense_1')(flatten)
    x = Dropout(0.4, name='dropout')(x)
    p = Dense(4, activation='softmax', name='softmax')(x)

    model = Model([x1_in, x2_in, x3_in], p)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])

    model.summary()
    print(">>>融合模型2加载结束。。。")

    return model


def create_model():
    print(">>>开始加载Bert模型。。。")
    bert_model = load_trained_model_from_checkpoint(config.bert_config_path, config.bert_checkpoint_path, trainable=False)

    # print(bert_model.summary())

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    # print("x's type = ", type(x))
    # print("x1 = ", x)
    # tensor_name = "functional_3/Encoder-12-FeedForward-Norm/add_1:0"
    # tf.print(tensor_name)
    # x = bert_model([x1_in, x2_in], name='my_bert_model')
    # x = Lambda(lambda x: x[:, :], name='last_layer')(x)
    # x = Lambda(lambda x: x[:, 0], name='last_layer')(x)  # 取出[CLS]对应的向量用来做分类
    x = Lambda(lambda x: x[:, 0], name='last_layer_1')(x)  # 取出[CLS]对应的向量用来做分类
    # x = Flatten()(x)
    # x = Dense(768)(x)
    p = Dense(2, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)

    '''
    inputs = bert_model.inputs[:2]
    dense = bert_model.get_layer('Encoder-12-FeedForward-Norm').output
    outputs = Dense(2, activation='softmax')(dense)
    model = Model(inputs, outputs)
    '''

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    print(">>>Bert模型加载结束。。。")
    model.summary()

    return model


def load_X3():
    x3 = np.array([1, 2, 3])
    X3 = np.array([x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3, x3])
    print("X3.shape = ", X3.shape)

    return X3


token_dict = {}
with codecs.open(config.bert_dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = bert_Tokenizer(token_dict)

# 测试3维融合模型
concat_model2 = create_concat_model2()

# 测试融合模型
concat_model = create_concat_model()
X3 = load_X3()
# print("X3 = ", X3)

texts, labels = get_texts()

X1, X2 = load_data(tokenizer, texts)
# print("X1 = ", X1)
print("X1.shape = ", X1.shape)
print("X2.shape = ", X2.shape)
print("labels = ", labels)
print("labels' length = ", len(labels))
Y = to_categorical(labels)

print("*" * 200)

# model = create_model()

print(">>>开始训练模型。。。")
concat_model.fit([X1, X2, X3], Y, epochs=1, batch_size=10, verbose=1)
# model.fit([X1, X2], Y, epochs=1, batch_size=10, verbose=1)

# model.summary()
# model_path = "result/tuned_bert_model_test.h5"
# model.save(model_path)

# old_model = load_model(model_path)
# print(old_model.summary())

print("*" * 200)
'''
print(">>>intermediate_layer_model...")
# layer_name = 'functional_3'
layer_name = 'last_layer'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(name=layer_name).output)
# intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(index=3).output)
intermediate_layer_model.summary()
layer_name2 = 'last_layer_1'

print("*" * 200)
# 测试
texts_test = get_texts_2()
'''
X1_test, X2_test = load_data(tokenizer, texts_test)

# Y_pred = model.predict([X1_test[0], X2_test[0]])
# print("Y_pred = ", Y_pred)
print("*" * 200)

text = '非常差！不会再去了！！'
tokens = tokenizer.tokenize(text)
indices, segments = tokenizer.encode(first=text, max_len=512)
# print("indices = ", indices[:10])
# print("segments = ", segments[:10])
'''
'''

'''
predicts = intermediate_layer_model.predict([X1_test, X2_test])[0]
# predicts = intermediate_layer_model.predict([np.array([indices]), np.array([segments])])
print("predicts's length = ", len(predicts))
print("predicts[0]'s length = ", len(predicts[0]))
print("tokens = ", tokens)
for i, token in enumerate(tokens):
    print("token = ", token, ", predicts = ", predicts[i].tolist()[:5])
    # print(token, predicts[i].tolist()[:5])
'''

print(">>>Here is what I need...")
for text_test in texts_test:
    tokens = tokenizer.tokenize(text_test)
    indices, segments = tokenizer.encode(first=text_test, max_len=512)
    print("indices = ", indices[:10])
    print("segments = ", segments[:10])
    predicts_test = intermediate_layer_model.predict([np.array([indices]), np.array([segments])])
    print("predicts_test_origin's length = ", len(predicts_test))
    predicts_test = predicts_test[0]
    print("predicts_test = ", predicts_test[:6])
    print("predicts_test's length = ", len(predicts_test))
    print("predicts_test[0]'s length = ", len(predicts_test[0]))
    break

'''
'''

'''
text = '我是中国人,他是日本人,你们呢'
tokens = tokenizer.tokenize(text)
print("tokens = ", tokens)
indices, segments = tokenizer.encode(first=text, max_len=512)
print("indices = ", indices)
print("indices' length = ", len(indices))
print("segments = ", segments)
'''


# model.fit()

# model.predict()

print(">>>end of absa_bert_tune.py...")


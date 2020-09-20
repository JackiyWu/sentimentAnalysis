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

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report, f1_score, accuracy_score

import config

# 读取训练集和测试集
train_df = pd.read_csv(config.baidu_sentiment, sep='\t', names=['id', 'type', 'contents', 'labels']).astype(str)
test_df = pd.read_csv(config.baidu_sentiment_test, sep='\t', names=['id', 'type', 'contents']).astype(str)
# test_df = pd.read_csv(config.baidu_sentiment_test, sep='\t', names=['id', 'type', 'contents'])[:3]
# print("train_df = ", train_df)
# print("train_df's type = ", type(train_df))
# print("test_df = ", test_df)
# print("test_df's type = ", type(test_df))

maxlen = 120  # 设置序列长度为120，要保证序列长度不超过512

# 预训练好的模型
config_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'config/keras_bert/chinese_L-12_H-768_A-12/vocab.txt'

# 将词表中的词编号转换为字典
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# 重写tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R


tokenizer = OurTokenizer(token_dict)


# 让每条文本的长度相同，用0填充
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


# data_generator只是一种为了节约内存的数据方式
class data_generator:
    def __init__(self, data, batch_size=16, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
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
                if len(X1) == self.batch_size or i == idxs[-1]:  # 如果当前输入数据已经够一个batch_size了or已经处理到了最后一条数据，那么就进行编码
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []  # 此处可以测试一下什么效果


# 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


# bert模型设置
def build_bert(nclass):
    # bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=768)  # 加载预训练模型
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),    # 用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


# 训练数据、测试数据和标签转化为模型输入格式
DATA_LIST = []
LABELS = []
for data_row in train_df.iloc[:].itertuples():
    # print("data_row = ", data_row)
    # print("data_row's type = ", type(data_row))
    DATA_LIST.append((data_row.contents, to_categorical(data_row.labels, 3)))
    LABELS.append(int(data_row.labels))
DATA_LIST = np.array(DATA_LIST)
# print("DATA_LIST = ", DATA_LIST)
# print("LABELS = ", LABELS)
# print("DATA_LIST's type = ", type(DATA_LIST))

DATA_LIST_TEST = []
for data_row in test_df.iloc[:].itertuples():
    DATA_LIST_TEST.append((data_row.contents, to_categorical(0, 3)))
DATA_LIST_TEST = np.array(DATA_LIST_TEST)


# 交叉验证训练和测试模型
def run_cv(nfold, data, data_labels, data_test, folds=1):
# def run_cv(nfold, data, data_labels, data_test, folds, ratio):
#     kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
    train_model_pred = np.zeros((len(data), 3))
    # print("train_model_pred = ", train_model_pred)
    test_model_pred = np.zeros((len(data_test), 3))

    # for i, (train_fold, test_fold) in enumerate(kf):
    for i in range(folds):
        # print("train_fold = ", train_fold)
        # print("train_fold's type = ", type(train_fold))
        # print("test_fold = ", test_fold)
        # print("test_fold's type = ", type(test_fold))
        # X_train, X_valid, = data[train_fold, :], data[test_fold, :]
        print("data's type = ", type(data))
        train_length = int(len(data) * 0.7)
        print("train_length = ", train_length)
        data = list(data)
        X_train = np.array(data)[:train_length, :]
        X_valid = np.array(data)[train_length:, :]
        # X_train, X_valid = data[:train_length, :], data[train_length:, :]
        print("X_train's type = ", type(X_train))

        y_val = data_labels[train_length:]
        y_val = np.array(y_val)

        model = build_bert(3)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)   # 早停法，防止过拟合
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)  # 当评价指标不在提升时，减少学习率
        checkpoint = ModelCheckpoint('./bert_dump/' + str(i) + '.hdf5', monitor='val_acc', verbose=2, save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)  # 验证集是从训练集中分出来的
        test_D = data_generator(data_test, shuffle=False)
        # 模型训练
        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=3,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )
        '''

        model.fit(train_D.__iter__(),
                  steps_per_epoch=len(train_D),
                  epochs=3,
                  validation_split=0.33,
                  # verbose=2,
                  callbacks=[early_stopping, plateau, checkpoint],)
        '''

        # model.load_weights('./bert_dump/' + str(i) + '.hdf5')

        # return model
        # train_model_pred = model.predict(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        train_model_pred = model.predict(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        # train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
        # print("train_model_pred = ", train_model_pred)
        # print("train_model_pred's type = ", type(train_model_pred))
        test_model_pred += model.predict(test_D.__iter__(), steps=len(test_D), verbose=1)

        del model
        gc.collect()   # 清理内存
        K.clear_session()   # clear_session就是清除一个session
        # break

    y_val_pred = np.argmax(train_model_pred, axis=1)

    # 准确率：在所有预测为正的样本中，确实为正的比例
    # 召回率：本身为正的样本中，被预测为正的比例
    # print("val_y = ", val_y)
    # print("y_val_pred = ", list(y_val_pred))
    precision, recall, fscore, support = score(y_val, y_val_pred)
    print("precision = ", precision)
    print("recall = ", recall)
    print("fscore = ", fscore)
    print("support = ", support)

    report = classification_report(y_val, y_val_pred, digits=4, output_dict=True)

    print(report)

    F1_score = f1_score(y_val_pred, y_val, average='macro')
    # F1_score = f1_score(y_val_pred, val_y, average='weighted')

    print('f1_score:', F1_score, 'ACC_score:', accuracy_score(y_val_pred, y_val))

    return train_model_pred, test_model_pred


# model = build_bert(3)
# n折交叉验证
train_model_pred, test_model_pred = run_cv(2, DATA_LIST, LABELS, DATA_LIST_TEST, 1)
# train_model_pred, test_model_pred = run_cv(1, DATA_LIST, LABELS, DATA_LIST_TEST, 1, 0.75)
# print("train_model_pred = ", train_model_pred)
# print("test_model_pred = ", test_model_pred)

test_pred = [np.argmax(x) for x in test_model_pred]

# 将测试集预测结果写入文件
output = pd.DataFrame({'id': test_df.id, 'sentiment': test_pred})
output.to_csv('result/bert_results.csv', index=None)

print("end of bert_demo2...")


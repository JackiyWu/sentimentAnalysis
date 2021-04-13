import pandas as pd
from aip import AipNlp
# import synonyms as syn
from pyecharts.charts import WordCloud
import collections
import numpy as np
import codecs
import csv
from snownlp import SnowNLP
import re
import sys
import string
from zhon.hanzi import punctuation
import zhon
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import xiangshi as xs

import auto_absa_config as config


file_names = ["big", "micro", "middle"]
# file_names = ["big", "micro", "middle", "middleBig", "small"]

punctuations = ['。', '？', '！', '，', '、', '；', '：', '“', '”', '（', '）', '-', '.', '《', '》', '）']
punc = string.punctuation + punctuation

stoplist = pd.read_csv('config/stopwords.txt').values

# 定义词性标签
noun = ['Ng', 'n', 'nr', 'ns', 'nt', 'nz']
verb = ['v', 'vd', 'vn', 'vg']
adj = ['Ag', 'a', 'ad', 'an']
adv = ['dg', 'd']
pron = ['r']
all_label = ['ng', 'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'vg', 'ag', 'a', 'ad', 'an', 'dg', 'd', 'r']


# 读取文件
def initData(file_path, debug=False):
    print("》》正在读取文件内容。。。")

    origin_data = produceAllData(file_path, debug)
    titles = origin_data['题目']
    contents = origin_data['内容']

    return origin_data, titles, contents


# 读取所有文件，生成一份数据
def produceAllData(path, debug):
    all_data = pd.DataFrame()

    for name in file_names:
        current_path = path + name + ".csv"
        current_data = produceOneData(current_path, debug)
        print(name, "data's shape = ", current_data.shape)

        all_data = all_data.append(current_data)

    # 索引重置
    all_data = all_data.reset_index(drop=True)

    return all_data


# 处理单个csv文件，生成指定格式
def produceOneData(path, debug=False):
    data = pd.read_csv(path)
    if debug:
        data = data[:5]

    # print(data['题目'])

    return data


# 计算题目和内容的相关性，输入题目和内容的DataFrame，返回相关性结果
def calculateSimilarity(titles, contents):
    length_tiltes = titles.shape[0]
    length_contents = contents.shape[0]
    if length_tiltes != length_contents:
        sys.exit(-1)

    result = []
    for i in range(length_tiltes):
        # print("title = ", titles[i])
        # print("content = ", contents[i])
        temp = xs.cossim([titles[i]], [contents[i]])
        result.append(temp)

    # result = pd.DataFrame(result)

    return result


# 计算内容的标点个数
def calculatePunctuationNumber(text):
    temp = 0
    for i in text:
        if i not in punc:
            continue
        if i in punctuations:
            temp += 1

    return temp


# 计算内容的句子数+平均句子长度
# 根据标点符号数
def calculateSentenceNumber(text):
    temp = 1
    if len(re.findall(zhon.hanzi.sentence, text)) > 1:
        temp = len(re.findall(zhon.hanzi.sentence, text))

    return temp, int(len(text) / temp)


# 词性标注
def taggingPartOfSpeech(text):
    noun = 0
    verb = 0
    adj = 0
    adv = 0
    pron = 0
    others = 0
    # tagging_result = {'noun': 0, 'verb': 0, 'adj': 0, 'adv': 0, 'others': 0, 'pron': 0}
    words = pseg.cut(text)
    for current_word in words:
        # print(current_word.word, current_word.flag)
        flag = str(current_word.flag)
        if flag not in all_label:
            others += 1
            # tagging_result['others'] += 1
        elif flag.lower().startswith('n'):
            noun += 1
            # tagging_result['noun'] += 1
        elif flag.lower().startswith('v'):
            verb += 1
            # tagging_result['verb'] += 1
        elif flag.lower().startswith('a'):
            adj += 1
            # tagging_result['adj'] += 1
        elif flag.lower().startswith('d'):
            adv += 1
            # tagging_result['adv'] += 1
        elif flag.lower().startswith('r'):
            pron += 1
            # tagging_result['pron'] += 1

    total = noun + verb + adj + adv + pron + others

    return float(noun / total), float(verb / total), float(adj / total), float(adv / total), float(pron / total), float(others / total)


# 将词性比例转为最终格式（每个词性所占比例）,顺序为：名词 动词 形容词 副词 代词 其他
def transformTagging(dicts):
    pass


# 题目和内容的词语重叠个数
def calculateOverlapWordsNumber(title, content):
    # print("*" * 160)
    # print("title:", title)
    # print("content:", content)
    # print(set(title) & set(content))

    return len(set(title) & set(content))


# 找到TopK特征词
def findTopKWords(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    words = vectorizer.get_feature_names()
    sort = np.argsort(tfidf.toarray(), axis=1)
    key_words = pd.Index(words)[sort].tolist()
    return key_words


# 保存主题结果
def saveTopKResult(keyWords, topK):

    # print("》》》打印每个topic的前10个关键词")
    length = len(keyWords)
    for i in range(length):
        print("Top", str(i), ":", keyWords[i][-topK:])


if __name__ == "__main__":
    print("》》》》开始提取特征。。。")

    # 1.读取文件
    origin_data, titles, contents = initData(config.feature_data)
    print("origin_data's columns = ", origin_data.columns.tolist())
    print("*" * 160)
    # print("titles", titles.head())
    # print("*" * 160)
    # print("contents", contents.head())
    # print("*" * 160)

    titles_cuted = []
    contents_cuted = []
    titles_cuted_nostop = []
    contents_cuted_nostop = []

    similarity = []
    punctuation_number = []  # 标点数量
    sentence_number = []  # 句子数量
    average_sentence_length = []  # 平均句子长度
    titles_word_number = []  # 题目词数
    contents_word_number = []  # 内容词数
    average_word_number = []  # 句子平均词数
    overlap_word_number = []  # 题目和句子重叠的单词数量
    titles_tagging_noun = []  # 题目词性个数
    titles_tagging_verb = []  # 题目词性个数
    titles_tagging_adj = []  # 题目词性个数
    titles_tagging_adv = []  # 题目词性个数
    titles_tagging_pron = []  # 题目词性个数
    titles_tagging_others = []  # 题目词性个数
    contents_tagging_noun = []  # 内容词性个数
    contents_tagging_verb = []  # 内容词性个数
    contents_tagging_adj = []  # 内容词性个数
    contents_tagging_adv = []  # 内容词性个数
    contents_tagging_pron = []  # 内容词性个数
    contents_tagging_others = []  # 内容词性个数

    # 遍历数据，最好只遍历一次完成所有事情
    length = len(titles)
    print("》》》正在遍历数据 计算各种指标。。。")
    for i in range(length):
        if i % 1500 == 0:
            print("》》当前已处理数据", i, "条")
        current_title = titles[i]
        current_content = contents[i]

        # 2.各种计算提取特征
        # ① 题目和内容相关性
        similarity.append(xs.cossim([current_title], [current_content]))

        # 对题目和内容进行词性标注
        # ⑥ 题目词性个数所占比例，分词
        noun, verb, adj, adv, pron, others = taggingPartOfSpeech(current_title)
        # print("noun = ", noun)
        # print("verb = ", verb)
        # print("adj = ", adj)
        # print("adv = ", adv)
        # print("pron = ", pron)
        # print("others = ", others)
        titles_tagging_noun.append(noun)
        titles_tagging_verb.append(verb)
        titles_tagging_adj.append(adj)
        titles_tagging_adv.append(adv)
        titles_tagging_pron.append(pron)
        titles_tagging_others.append(others)

        # ⑦ 内容词性个数，分词
        noun, verb, adj, adv, pron, others = taggingPartOfSpeech(current_content)
        contents_tagging_noun.append(noun)
        contents_tagging_verb.append(verb)
        contents_tagging_adj.append(adj)
        contents_tagging_adv.append(adv)
        contents_tagging_pron.append(pron)
        contents_tagging_others.append(others)

        # ② 标点个数
        punctuation_number.append(calculatePunctuationNumber(current_content))

        # ③ 内容句子数 + 平均句子长度
        current_sentence_number, current_average_sentence_length = calculateSentenceNumber(current_content)
        sentence_number.append(current_sentence_number)
        average_sentence_length.append(current_average_sentence_length)

        # 对题目和内容进行分词+去停用词
        current_title_cuted = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(current_title))  # 去标点符号
        current_title_cuted = jieba.lcut(current_title_cuted)  # 分词
        titles_cuted.append(current_title_cuted)
        current_title_cuted_nostop = [word.strip() for word in current_title_cuted if word not in stoplist]  # 去停用词
        titles_cuted_nostop.append(current_title_cuted_nostop)

        current_content_cuted = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(current_content))
        current_content_cuted = jieba.lcut(current_content_cuted)  # 分词
        contents_cuted.append(current_content_cuted)
        current_content_cuted_nostop = [word.strip() for word in current_content_cuted if word not in stoplist]  # 去停用词
        contents_cuted_nostop.extend(current_content_cuted_nostop)

        # ④ 统计题目词数，分词
        titles_word_number.append(len(current_title_cuted))

        # ⑤ 统计内容词数，分词
        contents_word_number.append(len(current_content_cuted))

        # ⑧ 内容平均句子词数，分词
        average_word_number.append(int(len(current_content_cuted) / current_sentence_number))

        # ⑨ 题目和内容的词语重叠个数，分词+去停用词
        overlap_word_number.append(calculateOverlapWordsNumber(current_title_cuted_nostop, current_content_cuted_nostop))

    print("similarity's length = ", len(similarity))
    print("punctuation_number's length = ", len(punctuation_number))
    print("sentence_number's length = ", len(sentence_number))
    # print("titles_cuted = ", titles_cuted)
    # print("titles_cuted_nostop = ", titles_cuted_nostop)
    # print("contents_cuted = ", contents_cuted)
    print("titles_word_number's length = ", len(titles_word_number))
    print("contents_word_number's length = ", len(contents_word_number))
    print("average_word_number's length = ", len(average_word_number))
    print("overlap_word_number's length = ", len(overlap_word_number))
    # print("titles_tagging's length = ", len(titles_tagging))
    # print("contents_tagging's length = ", len(contents_tagging))

    # 将词性比例转为最终格式（每个词性所占比例）

    # 将列表转为最终的DataFrame
    col_names = ['题目词数', '内容词数', '内容句子数', '题目-名词比例', '题目-动词比例', '题目-形容词比例', '题目-副词比例', '题目-代词比例', '题目-其他比例', '内容-名词比例',
                 '内容-动词比例', '内容-形容词比例', '内容-副词比例', '内容-代词比例', '内容-其他比例', '句子长度', '句子词数', '标点', '题目内容相似度', '题目内容词语重叠数']
    final_result = [titles_word_number, contents_word_number, sentence_number, titles_tagging_noun, titles_tagging_verb,
                    titles_tagging_adj, titles_tagging_adv, titles_tagging_pron, titles_tagging_others,
                    contents_tagging_noun, contents_tagging_verb, contents_tagging_adj, contents_tagging_adv,
                    contents_tagging_pron, contents_tagging_others, average_sentence_length, average_word_number,
                    punctuation_number, similarity, overlap_word_number]
    if len(col_names) != len(final_result):
        sys.exit(-2)

    col_number = len(col_names)
    # print("一共有", col_number, "列")
    df = pd.DataFrame()
    for i in range(col_number):
        origin_data[col_names[i]] = pd.Series(final_result[i])

    # print(origin_data)
    print("origin_data's columns = ", origin_data.columns.tolist())

    # 题目和内容单独保存
    titles_contents = pd.DataFrame()
    titles_contents['题目'] = pd.Series(titles)
    titles_contents['内容'] = pd.Series(contents)
    path1 = config.feature_data + "titles_contents.csv"
    titles_contents.to_csv(path1)

    # 3.结果保存至文件
    path = config.feature_data + "all_types.csv"
    origin_data = origin_data.drop('编号', 1)
    origin_data = origin_data.drop('题目', 1)
    origin_data = origin_data.drop('内容', 1)
    origin_data.to_csv(path)

    # ⑩ TopK特征词
    # print("contents_cuted_nostop's length = ", len(contents_cuted_nostop))
    contents_cuted_nostop = [' '.join(contents_cuted_nostop)]
    # print("contents_cuted_nostop's length = ", len(contents_cuted_nostop))
    keyWords = findTopKWords(contents_cuted_nostop)
    topK = 400
    print("Top", str(topK), ":", keyWords[0][-topK:])

    # topKs = [50, 100, 150, 200, 250, 300, 350, 400]
    # for topK in topKs:
    #     saveTopKResult(keyWords, topK)

    print("》》》》特征提取结束了哦。。。")


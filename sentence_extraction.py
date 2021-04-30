import pandas as pd
from aip import AipNlp
# import synonyms as syn
from pyecharts.charts import WordCloud
import collections
import absa_config as config
import numpy as np
import codecs
import csv
from snownlp import SnowNLP
import re
import paddlehub as hub
from cemotion import Cemotion as cet
import jieba

""" 你的 APPID AK SK """
APP_ID = '23894254'
API_KEY = 'cGjPl9Y96drYmhlM1aDgdtgV'
SECRET_KEY = 'i2UgIgvnyKvo2PEoT9dv89I3dh8QDuSS'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

options = dict()
options["type"] = 4

# 总的抱怨内容数量
total_number = 0

debug = False


def readAndExtract(text, threshold):
    global total_number
    print(">>>正在提取词对...")
    senta = hub.Module(name='senta_bilstm')
    c = cet()
    ans = []
    ans2 = []  # 保存摘要和观点词
    ans3 = []  # 保存所有摘要及其情感分
    len_text = len(text)
    # for i in range(1, 20):
    for i in range(0, int(len_text)):
        temp = {}
        # print("ans3 = ", ans3)
        if i % 100 == 0:
            print("正在处理第", i, "条评论哦...")
        current_text = text[i].replace(' ', '，').replace('…', '，')
        try:
            res = []
            res2 = set()
            result_temp = client.commentTag(current_text, options)

            # 把句子中的空格换为逗号 锅底水是农夫山泉矿泉水
            s = SnowNLP(current_text)
            # if len(s.summary()) < 5:
            #     print("s.summary() < 5 continue。。。")
            #     continue
            # print("current_text = ", current_text)
            # print("SnowNLP:", s.summary())
            # 去标点符号+判断情感极性
            for l in s.summary():
                l = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(l.strip()))
                polarity, score = judgeSentimentByPaddle(senta, [l], threshold, c)
                # if score > 0.5:
                #     print("正向情感。。。", l)
                if l not in temp.keys():
                    temp[l] = score
                if polarity != "positive":
                    res2.add(l)
                # else:
                #     print("positive:", l)

            # 补充逻辑：①SnowNLP与百度的一起；②如果当前摘要有空格，按照空格split
            # 用paddle判断文本情感极性

            error_code = 'error_code'
            while error_code in result_temp.keys():
                result_temp = client.commentTag(current_text, options)

            items = result_temp['items']
            baidu = []
            for ch in items:
                # print("ch:", ch)
                couple = (ch['prop'], ch['adj'])
                res.append(couple)
                # 用百度接口抽取摘要内容
                abstract = ch['abstract'].replace('<span>', '').replace('</span>', '').replace('<span></span>', '').replace('\n', '').replace(',', '').replace('.', '').strip()
                abstract = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(abstract))
                baidu.append(abstract)
                polarity, score = judgeSentimentByPaddle(senta, [abstract], threshold, c)
                # if score > 0.5:
                #     print("正向情感。。。", l)
                if abstract not in temp.keys():
                    temp[abstract] = score
                if polarity != "positive":
                    res2.add(abstract)
                # else:
                #     print("positive:", abstract)
                '''
                if len(ch['adj']) > 0 and len(ch['abstract']) < 30:
                    couple2 = (, ch['adj'])
                    res2.append(couple2)
                    print("BaiduAI:", ch['abstract'], "adj = ", ch['adj'])
                '''
            # print("Baidu:", baidu)
            # print("res2:", res2)
            # print("res2 = ", res2)

            ans.append(res)
            if len(res2) > 0:
                total_number += len(res2)
                for r in res2:
                    ans2.append([r.strip()])
                    # ans2.extend(list(res2))
            else:
                print("res2列表没有值。。。")
                print("current_text =", current_text)
            # print("*" * 50)
            # print("*" * 50)

        except KeyError as e:
            print("current_text =", current_text)
            print('KeyError:'+str(e))
            print(i)
            pass
        except Exception as e:
            print(str(e))

        for key, value in temp.items():
            ans3.append([key, value])

    print("ans2 = ", ans2)

    return ans, ans2, ans3


# 传入文本，返回情感极性
def judgeSentimentByPaddle(senta, text, threshold, c):
    positive_probs_c = c.predict(text[0])
    res = senta.sentiment_classify(texts=text)
    # print("结果= ", res)
    # result = res[0]['sentiment_key']
    # if result == "positive":
    #     print("text = ", text, ", positive_probs = ", res[0]['positive_probs'], ", negative_probs = ", res[0]['negative_probs'])

    if res[0]['negative_probs'] >= threshold or res[0]['sentiment_key'] != 'positive':
        if res[0]['negative_probs'] < 0.5 and positive_probs_c > 0.9:
            # print("paddle判定为负向,其概率为", str(res[0]['negative_probs']), ";Cemotion判定为正向，其概率为", str(positive_probs_c), ";text = ", text)
            return 'positive', 1 - res[0]['negative_probs']
        else:
            if 0.5 > res[0]['negative_probs'] >= threshold:
                return 'negative', 1 - res[0]['negative_probs']
                # print("threshold-", threshold, ": ", text)
                # print("paddle Cemotion同时判定为负向，且paddle判定为负向的概率为", res[0]['negative_probs'])

            return 'negative', 1 - res[0]['negative_probs']
    else:
        return 'positive', 1 - res[0]['negative_probs']


# 将方面词和观点词保存至文件
def saveToFile(result, result_combine, restaurant_name):
    print(">>>將result写入文件。。。")
    result = np.array(result)
    path1 = "result/aspect_viewpoint/" + restaurant_name + ".csv"
    with codecs.open(path1, "w") as f:
        writer = csv.writer(f)
        writer.writerows(result)
        f.close()

    result_combine = np.array(result_combine)
    path2 = "result/aspect_viewpoint/" + restaurant_name + "_combine.txt"
    with open(path2, 'w') as file_object:
        np.savetxt(file_object, result_combine, fmt='%s', delimiter=',')


# 将摘要写入文件
def saveAbstractToFile(result, debug):
    print(">>>将result写入文件...")
    # result = np.array(result)
    print("result = ", result)
    if debug:
        path = "result/farmer/complaints/abstracts_debug.csv"
    else:
        path = "result/farmer/complaints/abstracts.csv"
    # with codecs.open(path, 'w', newline='') as f:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)
        f.close()


# stoplist = pd.read_csv('config/stopwords_farmer.txt', error_bad_lines=False).values
# 对contents list去标点符号 分词 去停用词
def contentsProcess(contents):
    contents_cuted_nostop = []
    for current_content in contents:
        temp = []
        # print("before:", current_content)
        for current in current_content:
            current_content_cuted = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(current))
            current_content_cuted = jieba.lcut(current_content_cuted)  # 分词
            current_content_cuted_nostop = [word.strip() for word in current_content_cuted if word not in stoplist]  # 去停用词
            temp.append(''.join(current_content_cuted_nostop))
        # print("after:", temp)
        contents_cuted_nostop.append(temp)

    return contents_cuted_nostop


# 对摘要文本进行处理，去停用词
def processAbstracts(abstracts_list):
    abstracts_list = contentsProcess(abstracts_list)
    print("abstracts_list = ", abstracts_list)

    return abstracts_list


# 读取停用词
def getStopwords(path):
    stoplist = []
    f = open(path, 'r', encoding='utf-8')
    lines = f.readlines()
    for line in lines:
        stoplist.append(line.strip())

    return stoplist


# 将摘要写入文件
def saveAbstractToFile2(result, debug):
    print(">>>将result写入文件...")
    # result = np.array(result)
    print("result = ", result)
    if debug:
        path = "result/farmer/complaints/abstracts_all_debug.csv"
    else:
        path = "result/farmer/complaints/abstracts_all.csv"
    # with codecs.open(path, 'w', newline='') as f:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(result)
        f.close()


if __name__ == "__main__":
    print(">>>Begin...")

    print("读取停用词表")
    stoplist = getStopwords('config/stopwords_farmer.txt')
    print(stoplist)

    filepath = "datasets/farmer/farmer-negative.csv"

    data = pd.read_csv(filepath, sep=',')
    # data = pd.read_csv(filepath, sep=',', names=['id', 'reviews'])
    if debug:
        data = data[:8]
    text = data['reviews']
    # print("text = ", text.tolist())
    print("text's length = ", len(text))

    # 读取评论→抽取属性-观点词对
    threshold = 0.4
    ans, ans2, ans3 = readAndExtract(text, threshold)
    # print("ans = ", ans)
    print("ans3 = ", ans3)
    print("ans2's length = ", len(ans2))
    print("*" * 50)

    print("total_number = ", total_number)

    # 对摘要文本进行处理，去停用词
    ans2 = processAbstracts(ans2)

    # 保存ans2
    # saveAbstractToFile(ans2, debug)

    # 保存ans3
    saveAbstractToFile2(ans3, debug)

    print(">>>End...")


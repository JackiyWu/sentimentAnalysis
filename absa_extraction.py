import pandas as pd
from aip import AipNlp
import synonyms as syn
from pyecharts.charts import WordCloud
import collections
import absa_config as config
import numpy as np
import codecs
import csv
from snownlp import SnowNLP

""" 你的 APPID AK SK """
APP_ID = '23894254'
API_KEY = 'cGjPl9Y96drYmhlM1aDgdtgV'
SECRET_KEY = 'i2UgIgvnyKvo2PEoT9dv89I3dh8QDuSS'
client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

options = dict()
options["type"] = 4

# restaurant_names = ["kuaileai", "niuzhongniu", "shouergong", "xiaolongkan", "zhenghuangqi"]
# restaurant_names = ["chunla", "dingxiangyuan", "jialide", "jianshazui", "jiefu", "kuaileai", "niuzhongniu", "shouergong", "xiaolongkan", "zhenghuangqi"]
restaurant_names = ["zhenghuangqi", "xiaolongkan", "shouergong"]
# restaurant_names = ["kuaileai", "chunla"]

# 初始化情感词
none_negative_words = []


# 从情感词典中读取正向的情感词
def getPositiveWords():
    result = pd.read_csv(config.sentiment_dictionary_positive).values.tolist()
    # 转为一维列表
    result = sum(result, [])
    result = set(result)

    return result


def readAndExtract(text):
    print(">>>正在提取词对...")
    ans = []
    ans2 = []  # 保存摘要和观点词
    len_text = len(text)
    # for i in range(1, 20):
    for i in range(1, int(len_text)):
    #     print(i, "行:", text[i])
        try:
            res = []
            res2 = []
            print("current text = ", text[i])
            result_temp = client.commentTag(text[i], options)

            s = SnowNLP(text[i])
            print("SnowNLP:", s.summary())

            error_code = 'error_code'
            while error_code in result_temp.keys():
                result_temp = client.commentTag(text[i], options)

            items = result_temp['items']
            for ch in items:
                # print("ch:", ch)
                couple = (ch['prop'], ch['adj'])
                res.append(couple)
                if len(ch['adj']) > 0 and len(ch['abstract']) < 30:
                    couple2 = (ch['abstract'].replace('<span>', '').replace('</span>', '').replace('<span></span>', '').replace('\n', '').replace(',', '').replace('.', ''), ch['adj'])
                    res2.append(couple2)
                    print("BaiduAI:", ch['abstract'])
            ans.append(res)
            ans2.append(res2)
            # print("*" * 50)

        except KeyError as e:
            print("text[", i, "]=", text[i])
            print('KeyError:'+str(e))
            print(i)
            pass
        except Exception as e:
            print(str(e))

    return ans, ans2


def filterNotNegative(ans, data, positive_words, restaurant_name):
    print(">>>正在过滤非负向词对...")
    result = []
    result_combine = []

    len_ans = len(ans)
    for i in range(len_ans):
        temp = []
        # print("line:", data['reviews'][i+1])
        for couple in ans[i]:
            couple_aspect = couple[0]
            VIEWPOINT_IS_NOT_POSITIVE = True
            if len(couple) > 1:
                couple_viewpoint = couple[1]
                if couple_viewpoint in positive_words or couple_aspect in positive_words:
                    # print("couple_viewpoint = ", couple_viewpoint)
                    VIEWPOINT_IS_NOT_POSITIVE = False

            # 计算方面词与先验方面词的距离，分类
            aspect = calculateDistance(couple_aspect)
            Negative = data[aspect][i + 1] == 'Negative'
            if Negative and VIEWPOINT_IS_NOT_POSITIVE:
                temp.append(couple)
                result_combine.append(couple)
        result.append(temp)

    # 保存到文件
    saveToFile(result, result_combine, restaurant_name)

    return result, result_combine


# 将方面词和观点词保存至文件
def saveToFile(result, result_combine, restaurant_name):
    print(">>>將result寫入文件。。。")
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
def saveAbstractToFile(result, restaurant_name):
    print(">>>将result写入文件...")
    # result = np.array(result)
    print("result = ", result)
    path = "result/abstract/" + restaurant_name + ".csv"
    with codecs.open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(result)
        f.close()


def calculateDistance(word):
    aspect = 'location'
    distance = syn.compare(word, '位置')

    if distance < syn.compare(word, '服务'):
        aspect = 'service'
        distance = syn.compare(word, '服务')
    if distance < syn.compare(word, '价格'):
        aspect = 'price'
        distance = syn.compare(word, '价格')
    if distance < syn.compare(word, '环境'):
        aspect = 'environment'
        distance = syn.compare(word, '环境')
    if distance < syn.compare(word, '食物'):
        aspect = 'dish'

    return aspect


def plotWordCoulds(ans_combine, restaurant_name):
    print(">>>正在畫圖。。")
    word_counts = collections.Counter(ans_combine)
    word_counts_top_50 = word_counts.most_common(50)
    c = (
        WordCloud().add("", word_counts_top_50)  # 根据词频最高的词
            .render("result/wordsClouds/" + restaurant_name + "_50.html")  # 生成页面
    )

    word_counts_top_80 = word_counts.most_common(80)
    c = (
        WordCloud().add("", word_counts_top_80)  # 根据词频最高的词
            .render("result/wordsClouds/" + restaurant_name + "_80.html")  # 生成页面
    )

    word_counts_top_100 = word_counts.most_common(100)
    c = (
        WordCloud().add("", word_counts_top_100)  # 根据词频最高的词
            .render("result/wordsClouds/" + restaurant_name + "_100.html")  # 生成页面
    )

    word_counts_top_150 = word_counts.most_common(150)
    c = (
        WordCloud().add("", word_counts_top_150)  # 根据词频最高的词
            .render("result/wordsClouds/" + restaurant_name + "_150.html")  # 生成页面
    )

    word_counts_top_200 = word_counts.most_common(200)
    c = (
        WordCloud().add("", word_counts_top_200)  # 根据词频最高的词
            .render("result/wordsClouds/" + restaurant_name + "_200.html")  # 生成页面
    )


def filterNotNegative2(text, ans, positive_words, restaurant_name):
    print(">>>正在过滤非负向词对2...")
    result = []

    len_ans = len(ans)
    for i in range(len_ans):
        print("*" * 50)
        print(text[i + 1])
        temp = []
        for couple in ans[i]:
            couple_abstract = couple[0]
            if len(couple) > 1:
                couple_viewpoint = couple[1]
                if couple_viewpoint in positive_words:
                    print("摘要内容为正向:", couple)
                    continue
                else:
                    print("摘要内容为负向:", couple)
                temp.append(couple_abstract)
        # print("temp = ", temp)
        if len(temp) > 0:
            result.append(temp)
    # print("result = ", result)

    # 保存到文件
    saveAbstractToFile(result, restaurant_name)


if __name__ == "__main__":
    print(">>>Begin...")

    # 读取正向情感词
    positive_words = getPositiveWords()

    for restaurant_name in restaurant_names:
        print(">>>正在處理", restaurant_name)
        filepath = 'result/absa_20210327/negative/negative_predict_restaurant_' + restaurant_name + '.csv'

        data = pd.read_csv(filepath, sep=',', names=['ID', 'reviews', 'location', 'service', 'price', 'environment', 'dish'], encoding='utf-8')
        text = data['reviews']

        # 读取评论→抽取属性-观点词对
        ans, ans2 = readAndExtract(text)
        print("ans = ", ans)
        print("*" * 50)

        # 计算属性与先验属性距离，属性分类→过滤掉非负向属性-观点词对
        # ans, ans_combine = filterNotNegative(ans, data, positive_words, restaurant_name)
        # print("ans = ", ans)

        # 使用情感词典，过滤掉观点为正向的摘要
        filterNotNegative2(text, ans2, positive_words, restaurant_name)

        # plotWordCoulds(ans_combine, restaurant_name)

    print(">>>End...")


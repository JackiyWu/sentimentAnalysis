#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv
import codecs
import jieba
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

from sklearn.metrics import accuracy_score, precision_recall_fscore_support as score, classification_report

data_path = "datasets/usefulness/all_types_new.csv"

debug_data_length = 300

# 不需要标准化的属性
stay_aspects = ["空间评分", "动力评分", "操控评分", "能耗评分", "舒适性评分", "外观评分", "内饰评分", "性价比评分", "车辆类别", "上下班", "购物", "接送小孩",
                "自驾游", "跑长途", "商务差旅", "越野", "约会", "赛车", "拉货", "网约车", "组车队", "改装玩车", "昵称", "地点", "车辆匹配"]

# 结果保存路径
result_path = "result/auto_usefulness_0.csv"
result_path_debug = "result/auto_usefulness_debug.csv"

stoplist = pd.read_csv('config/stopwords.txt').values

# 原文本文件
contents_path = "datasets/usefulness/titles_contents.csv"

# 词典长度
dict_length = 150000  # 词典的长度

# Top400主题词
TopK = ['亮点', '本身', 'suv', '侧面', '精致', '加热', '电动车', '帅气', '很棒', '虚位', '中规中矩', '丰富', '几个', '最好', '副驾驶', '好像', '唯一', '距离', '必须', '到位', '充足', '转弯', '新款', '前面', '不如', '车头', '看过', '合资', '适应', '耐看', '不足', '30', '强劲', '整车', '15', '轴距', '导航', '媳妇', '尾灯', '按键', '能力', '出现', '样子', '进去', '不太', '好多', '路况', '最大', '稳定', '路段', '挺不错', '稳重', '霸气', '真是', '马力', '最高', '辅助', '开过', '想要', '范围', '认为', '搭配', '新能源', '白色', '值得', '过程', '拥挤', '力度', '反应', '看上去', '几乎', '中控台', '最终', '灵活', '提升', '经济', '技术', '十分', 'cc', '是因为', '有时候', '推背', '100', '磨合期', '全景', '上面', '估计', '年轻人', '以上', '自然', '影像', '灵敏', '同级', '还行', 'led', '胎噪', '18', '启动', '家人', '相比', '更好', '低速', '大概', '踩油门', '车辆', '路面', '属于', '每次', '压力', '停车', '档次', '25', '这点', '比较满意', '实在', '需求', '汽车', '长途', '减速带', '偶尔', '显示', '相对', '年轻', '吸引', '涡轮', '车门', '车里', '颠簸', '20t', '换挡', '不要', '控制', '代步', '担心', '预算', '显得', '希望', '材质', '黑色', '轮胎', '音响', '孩子', '马自达', '电动', '很快', '质感', '重要', '凯迪拉克', '平均', '十足', '方向', '头部', '城市', '一定', '路上', '腿部', '操作', 'xl', '本来', '更加', '舒适度', '车灯', '台车', '一种', '排量', '不算', '出去', '大众', '便宜', '百公里', '合适', '轿车', '不够', '经常', '颜色', '手机', '能够', '一台', '一辆', '科技', '正常', '味道', '减震', '本人', '轮毂', '天窗', '销售', '大家', '后期', '支撑', '后来', '系统', '开着', '雅阁', '还好', '不少', '发现', '身高', '习惯', '异味', '这辆', '影响', '外形', '简单', '丰田', '倒车', '符合', '位置', '要求', '奥迪', '实用', '顿挫', '一款', '体验', '手感', '平顺', '家里', '塑料', '用车', '下来', '线条', '调节', '性能', '综合', '容易', '没什么', '完美', '日常', '看到', '续航', '直接', '整个', '充电', '轻松', '漂亮', '提车', '精准', '异响', '奔驰', '4s店', '家用', '满足', '相当', '悬架', '风格', '中间', '个油', '后面', '时间', '坐在', '悬挂', '声音', '最后', '豪华', '差不多', '真皮', '省油', '老婆', '算是', '试驾', '情况', '基本上', '包裹', '用料', '造型', '变速箱', '保养', '这种', '宽敞', '当然', '迈锐宝', '对比', '看起来', '级车', '自动', '稍微', '车内', '两个', '原因', '不用', '上下班', '10', '中控', '速度', '只能', '亚洲', '使用', '肯定', '以后', '级别', '新车', '那种', '安全', '不到', '大灯', '凯美瑞', '尤其', '适合', '转向', '绝对', '提速', '价位', '加上', '时尚', '做工', '宝马', '隔音', '阿特', '行驶', '不能', '噪音', '需要', '超车', '颜值', '储物', '效果', '优惠', '油门', '感受', '里面', '当时', '这车', '考虑', '朋友', '大气', '品牌', '功能', '买车', '明显', '应该', '足够', '刹车', '模式', '东西', '主要', '知道', '接受', '表现', '之前', '平时', '基本', '目前', '乘坐', '车型', '底盘', '起步', '前排', '整体', '完全', '公里', '方便', '确实', '可能', '好看', '很大', '市区', '舒适', '开车', '够用', '舒适性', '运动', '加速', '空调', '左右', '后备箱', '地方', '发动机', '毕竟', '车身', '方面', '很多', '不会', '选择', '来说', '价格', '问题', '驾驶', '现在', '这款', '方向盘', '起来', '舒服', '配置', '操控', '特别', '设计', '性价比', '高速', '有点', '车子', '座椅', '喜欢', '后排', '满意', '内饰', '外观', '油耗', '非常', '动力', '空间']


# 从文件中读取数据
def readFromCSV(debug=False):
    data = pd.read_csv(data_path)
    if debug:
        data = data[:debug_data_length]

    # 获取X
    X = data.drop('发帖人_编号', 1)
    X = X.drop("支持", 1)

    X_need_standard = X.drop(stay_aspects, 1)
    X_stay = X[["空间评分", "动力评分", "操控评分", "能耗评分", "舒适性评分", "外观评分", "内饰评分", "性价比评分", "车辆类别", "上下班", "购物", "接送小孩", "自驾游", "跑长途", "商务差旅", "越野", "约会", "赛车", "拉货", "网约车", "组车队", "改装玩车", "昵称", "地点", "车辆匹配"]]

    # 数据标准化处理
    X_need_standard = dataPreprocess(X_need_standard)

    # 根据TopK主题词生成DataFrame
    X_topK = produceTopKDF(debug)
    print("X_topK's type = ", type(X_topK))

    print("X_topK's length = ", )
    if len(X_topK) != len(X_stay) or len(X_topK) != len(X_need_standard):
        print("三个DF长度不一致！！！！出错辣！！！！")

    X = pd.concat([X_stay, X_need_standard, X_topK], axis=1)
    print("X的列共有", len(X.columns.tolist()))
    X = X.to_numpy()

    # 获取Y
    data.loc[(data["支持"] <= 0), "支持"] = 0
    # data.loc[(0 < data["支持"]) & (data["支持"] <= 5), "支持"] = 1
    data.loc[(data["支持"] > 0), "支持"] = 1
    Y = data["支持"].to_numpy()

    return data, X, Y


# 数据预处理，归一化
def dataPreprocess(data):
    data = (data - data.mean()) / (data.std())

    return data


# 根据TopK主题词和原评论文本生成DataFrame
def produceTopKDF(debug=False):
    print("》》》正在生成TopK主题词的DataFrame。。。")
    titles_contents = pd.read_csv(contents_path)
    if debug:
        titles_contents = titles_contents[:debug_data_length]

    contents = titles_contents['内容'].tolist()
    contents = contentsProcess(contents)

    topK_DF = []

    for content in contents:
        temp_list = []
        for current_word in TopK:
            if current_word in content:
                temp_list.append(1)
            else:
                temp_list.append(0)
        topK_DF.append(temp_list)

    topK_DF = pd.DataFrame(topK_DF, columns=TopK)

    return topK_DF


# 读取评论文本内容
def readContents(debug=False):
    print("》》》正在读取评论文本内容。。。")
    titles_contents = pd.read_csv(contents_path)
    if debug:
        titles_contents = titles_contents[:debug_data_length]

    contents = titles_contents['内容'].tolist()
    contents = contentsProcess(contents)

    return contents


# 对contents list去标点符号 分词 去停用词
def contentsProcess(contents):
    contents_cuted_nostop = []
    for current_content in contents:
        current_content_cuted = re.sub("[\s+\.\!\/_,$%^*(+\"\'～]+|[+——！，。？?、~@#￥%……&*（）．；：】【|]+", "", str(current_content))
        current_content_cuted = jieba.lcut(current_content_cuted)  # 分词
        current_content_cuted_nostop = [word.strip() for word in current_content_cuted if word not in stoplist]  # 去停用词
        contents_cuted_nostop.append(current_content_cuted_nostop)

    return contents_cuted_nostop


# padding
def contentsPadding(contents):
    tokenizer = Tokenizer(num_words=dict_length)
    tokenizer.fit_on_texts(contents)

    word_index = tokenizer.word_index

    data_w = tokenizer.texts_to_sequences(contents)

    maxlen = calculate_maxlen(contents)
    data_T = sequence.pad_sequences(data_w, maxlen=maxlen)

    return data_T, word_index, maxlen


# 确定maxlen
def calculate_maxlen(texts):
    # 取评论长度的平均值+两个评论的标准差（假设评论长度的分布满足正态分布，则maxlen可以涵盖95左右的样本）
    lines_length = [len(line) for line in texts]
    lines_length = np.array(lines_length)

    maxlen = np.mean(lines_length) + 2 * np.std(lines_length)
    maxlen = int(maxlen)

    return maxlen


# 划分数据集
def dataSplit(X, Y, ratio):
    length = int(len(X) * ratio)
    X_train = X[: length]
    Y_train = Y[: length]
    X_validation = X[length:]
    Y_validation = Y[length:]

    print("X_train's length = ", len(X_train))
    print("Y_train's length = ", len(Y_train))
    print("X_validation's length = ", len(X_validation))
    print("Y_validation's length = ", len(Y_validation))

    return X_train, Y_train, X_validation, Y_validation


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


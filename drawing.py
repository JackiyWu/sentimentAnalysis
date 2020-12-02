# -*- coding:utf-8 -*-

import pandas as pd
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
    plt.xlim(xmin=-0.9, xmax=6.9)
    plt.ylim(ymin=0.715, ymax=0.772)
    plt.legend(fontsize=15.5, loc=1)  # 让图例生效
    plt.xticks(x, names, rotation=0)
    plt.tick_params(labelsize=16)
    plt.margins(6)
    plt.subplots_adjust(bottom=0.15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    # plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%1.1f'))
    # plt.xlabel(u"time(s)邻居")  # X轴标签
    # plt.ylabel("RMSE")  # Y轴标签
    # plt.title("A simple plot")  # 标题

    plt.show()


def read_csv(experiment_id):
    # data = pd.read_csv("C:\desktop\Research\选题相关\基于模糊系统和深度学习的在线评论情感计算\作图\\result\\result.csv")
    data = pd.read_csv("F:\Research\研究内容\一种融合模糊理论和深度学习的在线评论方面级情感分析\实验\\new_result.csv")
    data = data.loc[data["experiment_id"] == experiment_id]
    data['aspect'] = data['aspect'].astype(int)
    names = list(data['aspect'])
    print("names = ", names)
    y_F1 = list(data['macro_f1_score'])
    y_P = list(data['macro_precision'])
    y_R = list(data['macro_recall'])
    # y_ACC = list(data['acc_score'])

    return names, y_F1, y_P, y_R


if __name__ == "__main__":
    print("Begin at drawing...")

    names, y_F1, y_P, y_R = read_csv('filter')
    plot_test(names, y_F1, y_P, y_R)
    # print(names)
    # print(y_F1)

    print("End of drawing...")


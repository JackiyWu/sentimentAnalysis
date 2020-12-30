import pickle
import os
import codecs
import csv
import numpy as np
import absa_dataProcess as dp


# 读取文件
def readFromDictory(dictory, target_dictory):
    names_dict = {"chunla": [], "dingxiangyuan": [], "jialide": [], "jianshazui": [], "jiefu": [], "kuaileai": [], "niuzhongniu": [], "shouergong": [], "xiaolongkan": [], "zhenghuangqi": []}
    # 将文件夹中的文件按照餐厅name汇总
    for maindir, subdir, file_name_list in os.walk(dictory):
        print("1:", maindir)  # 当前主目录
        print("2:", subdir)  # 当前主目录下的所有目录
        print("3:", file_name_list)  # 当前主目录下的所有文件

        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径

            prefix = filename[0: filename.rfind('_', 1)]
            # print("prefix = ", prefix)
            if prefix not in names_dict.keys():
                names_dict[prefix] = []
            names_dict[prefix].append(apath)

    # print("names_dict.keys() = ", names_dict.keys())
    # print("names_dict.values() = ", names_dict.values())

    # 从文件中读取文本评论
    reviews_dict = {"chunla": [], "dingxiangyuan": [], "jialide": [], "jianshazui": [], "jiefu": [], "kuaileai": [], "niuzhongniu": [], "shouergong": [], "xiaolongkan": [], "zhenghuangqi": []}
    for key, values in names_dict.items():
        for file_path in values:
            with open(file_path, 'rb') as f:
                b = pickle.load(f)
                # print("b = ", b)

                for current_b in b:
                    if len(current_b.get('comment')) > 0:
                        reviews_dict[key].append(current_b.get('comment'))
                '''
                print(b[0].get('comment'))
                print("b[0] = ", b[0])
                b2dict = b[0]
                print("b2dict = ", b2dict)
                print("b2dict's type = ", type(b2dict))
                print(len(b2dict))
                '''
            # break
    # print("review_dict = ", reviews_dict)

    # 将字典中的评论数据存入文件
    write2file(reviews_dict, target_dictory)


# 写入文件
def write2file(reviews_dict, target_dictory):
    for key, values in reviews_dict.items():
        target_path = target_dictory + "/" + key + ".csv"
        # print("target_path = ", target_path)
        f = open(target_path, 'w', newline='', encoding='utf-8-sig')
        writer = csv.writer(f)

        i = 1
        for value in values:
            # value = np.array(value)
            print(i, ":", value)
            i += 1
            writer.writerow([value])

        f.close()


if __name__ == "__main__":
    print("this is the start of absa_readReview...")
    file_path = "F:\Research\DataSet\\restaurant reviews\dataUSE"
    target_path = "F:\Research\DataSet\\restaurant reviews\dataUSEReviews"

    # readFromDictory(file_path, target_path)
    '''
    test_string = "hello_world"
    print(test_string)
    test_string_split = test_string[0: test_string.rfind('_', 1)]
    print(test_string_split)
    '''

    X_validation, y_cols_validation, Y_validation = dp.getRestaurantDataByName('chunla')
    print(X_validation)

    print("this is the end of absa_readReview...")


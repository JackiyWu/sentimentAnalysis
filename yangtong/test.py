from sklearn.feature_extraction.text import CountVectorizer

print("hello world")

data_source = ["我 是 中国 人 中国 人 中国 人", "他 是 日本 人 吗 人 日本 他 他"]
str1 = "我 是 中国 人"
str1 = str1.split(" ")
print(str1)

result = []
for data in data_source:
    result.append(set(data.strip().split(" ")))
print("result = ", result)

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
#查看词频结果
print(X.toarray())
# 转置
print(X.toarray().T)


from keras_bert import extract_embeddings, Tokenizer, load_trained_model_from_checkpoint
import os
import codecs
import numpy as np

'''
 读取bert词向量
'''


config_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'config/keras_bert/chinese_L-12_H-768_A-12/vocab.txt'

os.environ['TF_KERAS'] = '1'

'''
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
# model.summary(line_length=120)

tokenizer = Tokenizer(token_dict)
text = '我是中国人,他是日本人,你们呢'
tokens = tokenizer.tokenize(text)
indices, segments = tokenizer.encode(first=text, max_len=512)
print("indices = ", indices[:10])
print("segments = ", segments[:10])

predicts = model.predict([np.array([indices]), np.array([segments])])[0]
for i, token in enumerate(tokens):
    print(token, predicts[i].tolist()[:5])

'''
model_path = "config/keras_bert/chinese_L-12_H-768_A-12"
print("*" * 100)

texts = ["清秀", "脏乱", "适得其反", "藏污纳垢"]

embeddings = extract_embeddings(model_path, texts)
print("")

# print("embeddings = ", embeddings)
print("embeddings' length = ", len(embeddings))

print("embeddings_0' length = ", len(embeddings[0][0]))
print("embeddings_1' length = ", len(embeddings[1][0]))

print("embeddings_0' type = ", type(embeddings[0][0]))
print("embeddings_0.shape = ", embeddings[0][0].shape)

print("embeddings_0 = ", list(embeddings[0][0]))
print("embeddings_1 = ", embeddings[1][0])

print(">>>end of bert_w2v...")


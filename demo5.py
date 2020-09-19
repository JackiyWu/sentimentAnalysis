from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
import numpy as np

config_path = 'C:/desktop/Coding/preWordEmbedding/albert/albert_zh-master/albert_config/albert_config_large.json'
checkpoint_path = 'C:/desktop/Coding/preWordEmbedding/albert/albert_zh-master/albert_config/albert_model.ckpt'
dict_path = 'C:/desktop/Coding/preWordEmbedding/albert/albert_zh-master/albert_config/vocab.txt'

token_dict = load_vocab(dict_path)
tokenizer = SimpleTokenizer(token_dict)
# 使用ALBERT
model = load_pretrained_model(config_path, checkpoint_path, albert=True)

# 编码测试
token_ids, segment_ids = tokenizer.encode(u'语言模型')
print(model.predict([np.array([token_ids]), np.array([segment_ids])]))


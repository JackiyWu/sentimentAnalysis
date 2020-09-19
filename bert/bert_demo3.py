
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths, load_trained_model_from_checkpoint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 预训练好的模型
config_path = '../config/keras_bert/bert_config.json'
checkpoint_path = '../config/keras_bert/bert_model.ckpt'
dict_path = '../config/keras_bert/vocab.txt'


if __name__ == "__main__":
    print(">>>begin in bert_demo3 ...")

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    print(bert_model.summary())

    print(">>>end of bert_demo3 ...")


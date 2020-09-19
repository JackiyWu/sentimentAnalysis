
from keras_bert import get_pretrained, PretrainedList, get_checkpoint_paths, load_trained_model_from_checkpoint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 预训练好的模型
config_path = 'C:\desktop\Coding\preWordEmbedding\chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'C:\desktop\Coding\preWordEmbedding\chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'C:\desktop\Coding\preWordEmbedding\chinese_L-12_H-768_A-12/vocab.txt'


if __name__ == "__main__":
    print(">>>begin in bert_demo3 ...")

    model_path = get_pretrained(PretrainedList.multi_cased_base)
    paths = get_checkpoint_paths(model_path)

    # 加载预训练模型
    bert_model = load_trained_model_from_checkpoint(
        config_file=paths.config,
        checkpoint_file=paths.checkpoint,
        training=False,
        trainable=True,
        use_task_embed=True,
        task_num=10,
    )

        # config_path,
        # checkpoint_path)

    print(">>>end of bert_demo3 ...")


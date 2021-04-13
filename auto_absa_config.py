#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 数据集
perception_data = "datasets/perception/"
perception_data_big = perception_data + "Big.csv"
perception_data_medium = perception_data + "Medium.csv"
perception_data_mediumBig = perception_data + "MediumBig.csv"
perception_data_micro = perception_data + "Micro.csv"
perception_data_small = perception_data + "Small.csv"
perception_data_new = perception_data + "New.csv"

# 有用性数据集
feature_data = "datasets/usefulness/"
feature_data_titles = []

# 评论属性
all_names = ["space", "space_label", "power", "power_label", "manipulation", "manipulation_label", "consumption",
           "consumption_label", "comfort", "comfort_label", "outside", "outside_label", "inside", "inside_label", "value", "value_label"]
aspect_names = ["space", "power", "manipulation", "consumption", "comfort", "outside", "inside", "value"]
col_names = ["space_label", "power_label", "manipulation_label", "consumption_label", "comfort_label", "outside_label", "inside_label", "value_label"]

# keras_bert
bert_path = "config/keras_bert/chinese_L-12_H-768_A-12"
bert_config_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_config.json'
bert_checkpoint_path = 'config/keras_bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
bert_dict_path = 'config/keras_bert/chinese_L-12_H-768_A-12/vocab.txt'


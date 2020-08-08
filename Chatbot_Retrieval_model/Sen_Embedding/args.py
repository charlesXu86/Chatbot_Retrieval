# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   args.py
 
@Time    :   2020/6/16 4:39 下午
 
@Desc    :
 
"""

import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.dirname(__file__)

model_dir = '/Data/public/Bert/chinese_wwm_L-12_H-768_A-12/'
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir =  '/home/xsq/nlp_code/Chatbot_Retrieval/Chatbot_Retrieval_model/Sen_Embedding/result/'
vocab_file = os.path.join(model_dir, 'vocab.txt')
# data_dir = os.path.join(model_dir, '../data/')

num_train_epochs = 10
batch_size = 128
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 5

# graph名字
graph_file = '/home/xsq/nlp_code/Chatbot_Retrieval/Chatbot_Retrieval_model/Sen_Embedding/result/graph'
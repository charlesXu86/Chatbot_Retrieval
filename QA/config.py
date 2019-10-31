# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   config.py
 
@Time    :   2019-10-29 17:35
 
@Desc    :
 
'''
import pathlib
import os

basedir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)

class Config():

    def __init__(self):
        self.bert_config_file = '/Data/public/Bert/chinese_L-12_H-768_A-12/bert_config.json'
        self.vocab_file = '/Data/public/Bert/chinese_L-12_H-768_A-12/vocab.txt'
        self.train_file = basedir + '/data/train-v1.1.json'
        self.output_dir = 'results'
        self.predict_file = basedir + '/data/dev-v1.1.json'
        self.init_checkpoint = '/Data/public/Bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
        self.do_lower_case = True
        self.verbose_logging = False
        self.master = None
        self.version_2_with_negative = False
        self.null_score_diff_threshold = 0.0
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_zone = None
        self.gcp_project = None
        self.num_tpu_cores = 8

        self.max_seq_length = 100
        self.doc_stride = 128
        self.max_query_length = 64


        self.do_train = True
        self.do_predict = True
        self.train_batch_size = 16
        self.predict_batch_size = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = 3.0
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.n_best_size = 20
        self.max_answer_length = 30

# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   config_bert.py
 
@Time    :   2019-10-29 17:35
 
@Desc    :
 
'''
import pathlib
import os

basedir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent)

class Config():

    def __init__(self):
        self.bert_config_file = '/Data/public/pretrained_models/tensorflow1.x/chinese-bert-wwm-ext/bert_config.json'
        self.vocab_file = '/Data/public/pretrained_models/tensorflow1.x/chinese-bert-wwm-ext/vocab.txt'
        self.data_dir = os.path.join(basedir, 'data/bert_sim/')
        self.output_dir = basedir + '/Chatbot_Retrieval_model_sss/Bert_sim/saved_model_bert'
        self.init_checkpoint = '/Data/public/pretrained_models/tensorflow1.x/chinese-bert-wwm-ext/bert_model.ckpt'

        self.do_lower_case = True
        self.verbose_logging = False
        self.master = None
        self.version_2_with_negative = False
        self.null_score_diff_threshold = 0.0
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_zone = None
        # self.gcp_project = None
        # self.num_tpu_cores = 8
        self.task_name = 'sim'
        self.gpu_memory_fraction = 0.8
        self.max_seq_length = 128

        self.do_train = True
        self.do_predict = False
        self.batch_size = 64
        self.learning_rate = 5e-5
        self.num_train_epochs = 3.0
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 1000
        # self.iterations_per_loop = 1000
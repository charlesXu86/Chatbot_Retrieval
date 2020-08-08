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
        self.bert_config_file = '/Data/public/Bert/chinese_L-12_H-768_A-12/bert_config.json'
        self.vocab_file = '/Data/public/Bert/chinese_L-12_H-768_A-12/vocab.txt'
        self.data_dir = os.path.join(basedir, 'data/dialogue_rewrite/corpus.txt')
        self.output_dir = basedir + '/Chatbot_Retrieval_model/Bert_sim/results'
        self.predict_file = basedir + '/data/bert_sim/dev.txt'
        self.test_file = basedir + '/data/bert_sim/test.txt'
        self.init_checkpoint = '/Data/public/Bert/chinese_L-12_H-768_A-12/bert_model.ckpt'

        self.train_checkpoint = '/home/xsq/nlp_code/Chatbot_Retrieval/Bert_sim/results'
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
        self.task_name = 'sim'
        self.gpu_memory_fraction = 0.8

        self.max_seq_length = 128
        self.doc_stride = 128
        self.max_query_length = 64


        self.do_train = False
        self.do_predict = True
        self.batch_size = 20
        self.predict_batch_size = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = 3.0
        self.warmup_proportion = 0.1
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.n_best_size = 20
        self.max_answer_length = 30
        self.eval_batch_size = 16
        # self.do_eval = False

        self.mode = 'train'
        self.encoder_type = 'bi'
        self.hidden_dim = 256
        self.emb_dim = 128
        self.max_enc_steps = 50    # max timesteps of encoder (max source text tokens)
        self.max_dec_steps = 30    # max timesteps of decoder (max summary tokens)
        self.beam_size = 4
        self.min_dec_steps = 5
        self.single_pass = False
        self.learning_rate = 0.15
        self.adagrad_init_acc = 0.1
        self.rand_unif_init_mag = 0.02
        self.trunc_norm_init_std = 1e-4
        self.max_grad_norm = 2.0
        self.pointer_gen = True
        self.coverage = False
        self.cov_loss_wt = 1.0
        self.convert_to_coverage_model = False
        self.restore_best_model = False
        self.debug = False
        self.log_root = './log'
        self.exp_name = 'extractive'
        self.vocab_size = 28000



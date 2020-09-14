# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   dialogue_predict.py
 
@Time    :   2019-11-06 14:25
 
@Desc    :
 
'''

import pathlib
import os
import tensorflow as tf
import numpy as np

# from Chatbot_Retrieval_model.QA.FAQ import FAQ
from Chatbot_Retrieval_model.QA.FAQ_v2 import FAQ
# from Chatbot_Retrieval_model.Bert_sim.run_similarity_bert import BertSim   # Bert语义相似度
# from Chatbot_Retrieval_model.Domain.domain_classifier_v2 import DomainCLS   # Domain 分类
from Chatbot_Retrieval_model.util.logutil import Logger

loginfo = Logger('FAQ_log', 'info')

# bs = BertSim()
# bs.set_mode(tf.estimator.ModeKeys.PREDICT)

# dc = DomainCLS()
# dc.set_mode(tf.estimator.ModeKeys.PREDICT)

def get_anwser(msg):

    resul = {
        'domain':'',
        'anwser':'',
    }

    robot = FAQ(usedVec=False)

    normal_query, anwser = robot.answer(msg, 'simple_pos')

    # sen2 = '我想买保险'
    # predict = bs.predict(msg, sen2)
    # result = predict[0][1]

    # domain_result = dc.predict(msg)
    # resul['domain'] = domain_result
    resul['user_query'] = msg
    resul['query'] = normal_query[0]
    resul['anwser'] = anwser

    return resul

def estimate_answer(candidate, answer):
    '''
    评估答案，暂时不用
    :param candidate:
    :param answer:
    :return:
    '''
    candidate = candidate.strip().lower()
    answer = answer.strip().lower()
    if candidate == answer:
        return True

    if not answer.isdigit() and candidate.isdigit():
        candidate_temp = "{:.5E}".format(int(candidate))
        if candidate_temp == answer:
            return True
        candidate_temp == "{:.4E}".format(int(candidate))
        if candidate_temp == answer:
            return True

    return False


# msg = '预算14万内买什么车好呢？'
# get_anwser(msg)

# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   Get_domain.py

@Time    :   2019-12-11 11:43

@Desc    :   获取domain预测的结果

'''

import tensorflow as tf
from Chatbot_Retrieval_model.Domain.domain_classifier_v2 import DomainCLS   # Domain 分类

dc = DomainCLS()
dc.set_mode(tf.estimator.ModeKeys.PREDICT)


def get_domain_res(msg):
    '''

    :param msg:
    :return:
    '''
    resul = {
        'domain': ''
         }

    domain_result = dc.predict(msg)
    resul['domain'] = domain_result

    return resul



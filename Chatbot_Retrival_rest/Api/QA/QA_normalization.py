# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   QA_normalization.py
 
@Time    :   2020/10/3 12:15 下午
 
@Desc    :   检索数据后处理
 
"""
from Chatbot_Retrieval_model.QA.es_qa import Searcher

es = Searcher()


def qa_normal(message):
    """

    """
    response = es.search_es(message)
    responses_sorted = sorted(response, key=lambda x: x['score'], reverse=True)
    answer = responses_sorted[0]['answers']

    return answer
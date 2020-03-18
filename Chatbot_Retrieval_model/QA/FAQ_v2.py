# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   FAQ.py
 
@Time    :   2019-10-30 14:26
 
@Desc    :   答案的语义匹配,修改数据源
 
'''

import os
import logging
import jieba
import json
import jieba.posseg as pseg
import tensorflow as tf
from collections import deque
from Chatbot_Retrieval_model.QA.ConnRedis import RedisDatabase

# from Chatbot_Retrieval_model.Bert_sim.run_similarity import BertSim

from Chatbot_Retrieval_model.QA.utils import (get_logger, similarity)

jieba.dt.tmp_dir = "./"
jieba.default_logger.setLevel(logging.ERROR)
logger = get_logger('faqrobot', logfile="faqrobot.log")


# 初始化语义相似度匹配模型
# bs = BertSim()
# bs.set_mode(tf.estimator.ModeKeys.PREDICT)

db = RedisDatabase()

class zhishiku():
    def __init__(self, q):
        self.q = [q]
        self.a = ""
        self.sim = 0
        self.q_vec = []
        self.q_word = []

    def __str__(self):
        return 'q=' + str(self.q) + '\na=' + str(self.a) + '\nq_word=' + str(self.q_word) + '\nq_vec=' + str(self.q_vec)

class FAQ(object):
    def __init__(self, zhishitxt=None, lastTxtLen=10, usedVec=True):
        # usedVec 如果是True 在初始化时会解析词向量，加快计算句子相似度的速度
        self.lastTxt = deque([], lastTxtLen)
        self.zhishitxt = zhishitxt
        self.usedVec = usedVec
        self.reload()

    def load_qa(self):
        '''
        加载问答知识库，连接redis
        :return:
        '''
        # print('问答知识库开始载入')
        self.zhishiku = []
        # with open(self.zhishitxt, 'r') as f:
        #     data = json.load(f)
        #     abovetxt = 0    # 上一行的种类： 0空白/注释  1答案   2问题
        #     for item in data:
        #         question = item['ques']
        #         anwser = item['anwser']
        #         self.zhishiku.append(zhishiku(question))
        #         self.zhishiku[-1].a += anwser
        #         print(question)
        data = db.r.lrange('faq', 0, -1)
        for item in data:
            item = json.loads(item)
            question = item['question']
            anwser = item['anwser']
            self.zhishiku.append(zhishiku(question))
            self.zhishiku[-1].a += anwser

        for t in self.zhishiku:
            for question in t.q:
                t.q_word.append(set(jieba.cut(question)))


    def load_embedding(self):
        '''
        加载bert语义匹配
        :return:
        '''
        from Chatbot_Retrieval_model.Bert_sim.run_similarity_bert import BertSim

        self.vecModel = BertSim()

    def reload(self):
        self.load_qa()
        self.load_embedding()

        # print('问答知识库载入完毕')

    def maxSimTxt(self, intxt, simCondision=0.15, simType='simple'):
        '''
        找出知识库里的和输入句子相似度最高的句子

        :param intxt: 输入文本
        :param simCondision: 相似度阈值
        :param simType:
        :return:
        '''
        self.lastTxt.append(intxt)
        if simType not in ('simple', 'simple_pos', 'vec'):
            return 'error:  maxSimTxt的simType类型不存在: {}'.format(simType)

        # 如果没有加载词向量，那么降级成 simple_pos 方法
        embedding = self.vecModel
        if simType == 'vec' and not embedding:
            simType = 'simple_pos'

        for t in self.zhishiku:
            questions = t.q_vec if simType == 'vec' else t.q_word
            in_vec = jieba.lcut(intxt) if simType == 'simple' else pseg.lcut(intxt)

            t.sim = max(
                similarity(in_vec, question, method=simType, embedding=embedding)
                for question in questions
            )
        maxSim = max(self.zhishiku, key=lambda x: x.sim)
        logger.info('maxSim=' + format(maxSim.sim, '.0%'))

        if maxSim.sim < simCondision:
            return '抱歉，我没有理解您的意思。请您询问有关汽车的话题。'

        return maxSim.a

    def answer(self, intxt, simType='simple'):
        """simType=simple, simple_POS, vec, all"""
        if not intxt:
            return ''

        if simType == 'all':  # 用于测试不同类型方法的准确度，返回空文本
            for method in ('simple', 'simple_pos', 'vec'):
                outtext = 'method:\t' + self.maxSim(intxt, simType=method)
                print(outtext)

            return ''
        else:
            outtxt = self.maxSimTxt(intxt, simType=simType)
            # 输出回复内容，并计入日志
        return outtxt


if __name__ == '__main__':
    data = '/home/xsq/nlp_code/Chatbot_Retrieval/data/FAQ/FAQ_baoxian.json'
    robot = FAQ(data, usedVec=False)
    while True:
        print('回复：' + robot.answer(input('输入：'), 'simple_pos') + '\n')
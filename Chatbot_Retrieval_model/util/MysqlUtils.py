# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   MysqlUtils.py
 
@Time    :   2020/6/12 3:13 下午
 
@Desc    :   数据查询
             用到的表（离线表）： dialog_control_log_detail， intent_feedback， dialog_control_log， acs_speech_item
             Mysql表： report_calculate_data_result
"""

import pymysql
import logging


logger = logging.getLogger(__name__)


class FAQData():

    def __init__(self):

        # 本地
        self.database = 'chatbot_cn'
        self.host = '172.18.86.20'
        self.username = 'user_rw'
        self.password = 'mysqlreadwrite'
        self.db = pymysql.connect(host=self.host, port=3306, user=self.username, passwd=self.password, db=self.database)

    def get_faq_data(self):
        logger.info('Connect mysql {} successfully'.format(self.db))
        cursor = self.db.cursor()
        cursor.execute("select id,"
                       "question,"
                       "answer,"
                       "category"
                       " from Chatbot_Retrieval_model_faq ")
        res = cursor.fetchall()
        logger.info('All data length is {}'.format(len(res)))
        # cursor.close()
        # self.db.close()

        return res


if __name__ == "__main__":
    speechId = '569353302990019129'
    FAQData().get_faq_data()
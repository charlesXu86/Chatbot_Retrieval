# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   RedisUtils.py
 
@Time    :   2020/3/12 9:40 上午
 
@Desc    :   pyrhon 连接redis
 
'''

import redis
import json

import logging

logger = logging.getLogger(__name__)


class RedisDatabase():

    def __init__(self):
        # self.host = '172.17.41.235'
        # self.port = 7404
        # self.passwd = 'RTRgHKD2RmurhTWR'
        # self.host = '124.70.194.208'
        self.host = '172.18.86.20'     # 测试服务器
        self.port = 6379
        self.r = redis.StrictRedis(host=self.host, port=self.port)
        logger.info('Connect Redis success {}'.format(self.r))

    def insert(self, keyName, jsonStr):
        '''
        插入数据
        '''
        self.r.lpush(keyName, jsonStr)


db = RedisDatabase()


def add_data():
    faq = {}
    faq['id'] = 1111
    faq['question'] = '你好'
    faq['answer'] = '您好，我是智能助手，请问有什么可以帮您。'

    if db.r.exists('chatbot'):
        db.r.delete('chatbot')
    db.insert(keyName='Chatbot_Retrieval_model', jsonStr=json.dumps(faq, indent=4, ensure_ascii=False))
    print('Success')


def get_data():
    result = db.r.lrange('P_ACS_NEVER_MORE:464812905667379226', 0, -1)
    print(result)


if __name__ == '__main__':
    db = RedisDatabase()
    add_data()
    # get_data()

#  redis-py使用connection pool来管理对一个redis server的所有连接，避免每次建立、释放连接的开销。默认，每个Redis实例都会维护一个自己的连接池。可以直接建立一个连接池，然后作为参数Redis，这样就可以实现多个Redis实例共享一个连接池。
# pool = redis.ConnectionPool(host='172.18.103.43', port=6379)

# r = redis.Redis(connection_pool=pool)
# r.set('foo', 'Bar')
# print(r.get('foo'))

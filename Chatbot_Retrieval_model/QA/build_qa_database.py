# -*- coding: utf-8 -*-
"""
   File Name：     build_qa_database
   Description :  在ES中构建问答库
   Author :       MeteorMan
   date：          2019/12/2

"""

import os
import time
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


class ProcessIntoES(object):
    def __init__(self, ip="192.168.8.183"):
        self._index = "law_qa_test_1"  # 相当于创建的MySQL数据库名称
        # 无用户名密码状态
        self.es = Elasticsearch([ip], port=9200)
        # 用户名密码状态
        # self.es = Elasticsearch([ip], http_auth=('admin', '123456'), port=9200)
        self.doc_type = "qa"  # 相当于在指定数据库中创建的表名称
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        # self.music_file = os.path.join(cur, 'data/qa_corpus.json')
        self.music_file = '/data/xiaobensuan/Codes/Chatbot_Retrieval/data/qa_corpus.json'

    '''创建ES索引，确定分词类型'''
    '''
    两种分词器使用的最佳实践是：索引时用ik_max_word，在搜索时用ik_smart。
    即：索引时最大化的将文章内容分词，搜索时更精确的搜索到想要的结果。
    '''

    def create_mapping(self):
        node_mappings = {
            "mappings": {
                self.doc_type: {  # type
                    "properties": {
                        "question": {  # field: 问题
                            "type": "text",  # lxw NOTE: cannot be string
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                            "index": "true"  # The index option controls whether field values are indexed.
                        },
                        "category": {  # field: 类别
                            "type": "text",
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                            "index": "true"
                        },
                        "answers": {  # field: 答案
                            "type": "text",  # lxw NOTE: cannot be string
                            "analyzer": "ik_max_word",
                            "search_analyzer": "ik_smart",
                            "index": "true"  # The index option controls whether field values are indexed.
                        },
                    }
                }
            }
        }
        if not self.es.indices.exists(index=self._index):
            self.es.indices.create(index=self._index, body=node_mappings, ignore=[400, 409])
            print("Create {} mapping successfully.".format(self._index))
        else:
            print("index({}) already exists.".format(self._index))

    '''批量插入数据'''

    def insert_data_bulk(self, action_list):
        success, _ = bulk(self.es, action_list, index=self._index, raise_on_error=True)
        print("Performed {0} actions. _: {1}".format(success, _))


'''初始化ES，将数据插入到ES数据库当中'''


def init_ES():
    pie = ProcessIntoES()
    # 创建ES的index
    pie.create_mapping()
    start_time = time.time()
    index = 0
    count = 0
    action_list = []
    BULK_COUNT = 1000  # 每BULK_COUNT个句子一起插入到ES中

    for line in open(pie.music_file, 'r', encoding='utf-8'):
        if not line:
            continue
        item = json.loads(line)
        index += 1
        action = {
            "_index": pie._index,
            "_type": pie.doc_type,
            "_source": {
                "question": item['question'],
                "category": item['category'],
                "answers": '\n'.join(item['answers']),
            }
        }
        action_list.append(action)
        if index > BULK_COUNT:
            pie.insert_data_bulk(action_list=action_list)
            index = 0
            count += 1
            print("bulk {} writted finished!".format(count))
            action_list = []
    end_time = time.time()

    print("Time cost:{0}".format(end_time - start_time))


if __name__ == "__main__":
    # 将知识库文件库插入到elasticsearch当中
    init_ES()

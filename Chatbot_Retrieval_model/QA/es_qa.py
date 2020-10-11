# -*- coding: utf-8 -*-
"""
   File Name：     law_qa
   Description :  ES检索式问答
   Author :       MeteorMan
   date：          2019/12/2

"""

from elasticsearch import Elasticsearch


class Searcher(object):
    def __init__(self, ip="192.168.8.183"):
        self._index = "law_qa_test_1"  # 相当于创建的MySQL数据库名称
        # 无用户名密码状态
        self.es = Elasticsearch([ip], port=9200)
        # 用户名密码状态
        # self.es = Elasticsearch([ip], http_auth=('admin', '123456'), port=9200)
        self.doc_type = "qa"  # 相当于在指定数据库中创建的表名称

    '''根据question进行事件的匹配查询'''

    def search_specific(self, ques, key="question"):
        query_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                key: ques
                            }
                        }
                    ],
                    "must_not": [],
                    "should": []
                }
            },
            "from": 0,
            "size": 20,
            "sort": [],
            "aggs": {}
        }
        '''
        Note：在这里返回的检索结果数量有两种设置方式：
        1.在es.search()中指定size参数值
        2.在query_body()中的size对应字段
        '''
        # searched = self.es.search(index=self._index, doc_type=self.doc_type, body=query_body, size=20)
        searched = self.es.search(index=self._index, doc_type=self.doc_type, body=query_body)
        # 输出查询到的结果
        return searched["hits"]["hits"]

    '''基于ES的问题查询'''

    def search_es(self, question):
        answers = []
        res = self.search_specific(question, 'question')
        for hit in res:
            answer_dict = {'score': hit['_score'] / 100, 'sim_question': hit['_source']['question'],
                           'answers': hit['_source']['answers'].split('\n')}
            answers.append(answer_dict)
        return answers


def main_qa():
    searcher = Searcher()
    # question = '我老公要起诉离婚 我不想离婚怎么办'
    while True:
        question = input('query:')
        responses = searcher.search_es(question)
        # print(responses)
        responses_sorted = sorted(responses, key=lambda x: x['score'], reverse=True)
        answer = responses_sorted[0]['answers']
        print('answer: ', answer)


if __name__ == "__main__":
    # 检索式问答
    main_qa()

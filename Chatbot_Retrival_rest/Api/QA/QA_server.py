# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   payback_class_controller.py

@Time    :   2019-06-10 14:44

@Desc    :  还款意愿分类接口封装

'''

from django.http import JsonResponse
import json
import logging
import datetime

from Chatbot_Retrieval_model.QA.dialogue_predict import get_anwser

logger = logging.getLogger(__name__)


def qa_server(request):
    if request.method == 'POST':

        # try:
            jsonData = json.loads(request.body.decode('utf-8'))
            msg = jsonData["msg"]
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = get_anwser(msg)
            print(result)
            dic = {
                "desc": "Success",
                "ques": msg,
                "result": result,
                "time": localtime
            }
            log_res = json.dumps(dic, ensure_ascii=False)
            logger.info(log_res)
            return JsonResponse(dic)
        # except Exception as e:
        #     logger.info(e)
    else:
        return JsonResponse({"desc": "Bad request"}, status=400)
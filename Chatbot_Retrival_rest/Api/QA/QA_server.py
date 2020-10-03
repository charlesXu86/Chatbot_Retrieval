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

from Chatbot_Retrival_rest.Api.QA.QA_normalization import qa_normal

logger = logging.getLogger(__name__)


def qa_server(request):
    if request.method == 'POST':

        try:
            jsonData = json.loads(request.body.decode('utf-8'))
            message = jsonData["message"]
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = qa_normal(message)
            dic = {
                "desc": "Success",
                "ques": message,
                "result": result,
                "time": localtime
            }
            log_res = json.dumps(dic, ensure_ascii=False)
            logger.info(log_res)
            return JsonResponse(dic)
        except Exception as e:
            logger.info(e)
    else:
        return JsonResponse({"desc": "Bad request"}, status=400)

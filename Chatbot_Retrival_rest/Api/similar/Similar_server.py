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

from Chatbot_Retrival_rest.Api.similar.Get_similar import get_similar_res_bert
# from Chatbot_Retrival_rest.Api.similar.Get_similar_albert import get_similar_res_albert

from Chatbot_Retrival_rest.Api.utils.LogUtils import Logger

logger = logging.getLogger(__name__)


def sim_server(request):
    if request.method == 'POST':

        try:
            jsonData = json.loads(request.body.decode('utf-8'))
            sen1 = jsonData["msg1"]
            sen2 = jsonData["msg2"]
            model = jsonData["model"]
            localtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            result = get_similar_res_bert(sen1, sen2)
            dic = {
                "desc": "Success",
                "ques1": sen1,
                "ques2": sen2,
                "model":  model,
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
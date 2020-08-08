# -*- coding: utf-8 -*-

"""
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   main.py
 
@Time    :   2020/7/18 7:46 下午
 
@Desc    :
 
"""

import web
from Chatbot_Retrieval_model.wx.handle import Handle

urls = (
    '/chatbot', 'Handle',
)

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()
# # -*- coding: utf-8 -*-
#
# '''
# @Author  :   Xu
#
# @Software:   PyCharm
#
# @File    :   Get_domain.py
#
# @Time    :   2019-12-11 11:43
#
# @Desc    :   获取domain预测的结果
#
# '''
#
# import tensorflow as tf
# # from Chatbot_Retrieval_model.Bert_sim.run_similarity_bert import BertSim
# # from Chatbot_Retrieval_model.Bert_sim.run_similarity_albert import AlBertSim
#
# # dc = BertSim()
# # dc.set_mode(tf.estimator.ModeKeys.PREDICT)
#
# # al_bs = AlBertSim()
# # al_bs.set_mode(tf.estimator.ModeKeys.PREDICT)
#
#
# def get_similar_res_bert(sen1, sen2):
#     '''
#
#     :param msg:
#     :return:
#     '''
#     resul = {
#         'sim_prob': ''
#          }
#
#     sim_result = dc.predict(sen1, sen2)
#     resul['sim_prob'] = float(sim_result[0][1])
#
#     return resul
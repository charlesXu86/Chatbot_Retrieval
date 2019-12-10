# -*- coding: utf-8 -*-

'''
@Author  :   Xu
 
@Software:   PyCharm
 
@File    :   model_metrics.py
 
@Time    :   2019-12-10 09:41
 
@Desc    :
 
'''

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

smoothing_function = SmoothingFunction().method4

class Metrics(object):
    def __init__(self):
        pass

    @staticmethod
    def bleu_score(references, candidates):
        """
        计算bleu值
        :param references: 实际值, list of string
        :param candidates: 验证值, list of string
        :return:
        """
        # 遍历计算bleu
        bleu1s = []
        bleu2s = []
        bleu3s = []
        bleu4s = []
        for ref, cand in zip(references, candidates):
            ref_list = [list(ref)]
            cand_list = list(cand)
            bleu1 = sentence_bleu(ref_list, cand_list, weights=(1, 0, 0, 0), smoothing_function=smoothing_function)
            bleu2 = sentence_bleu(ref_list, cand_list, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
            bleu3 = sentence_bleu(ref_list, cand_list, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing_function)
            bleu4 = sentence_bleu(ref_list, cand_list, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
            # print ("ref: %s, cand: %s, bleus: %.3f, %.3f, %.3f, %.3f"
            #        % (ref, cand, bleu1, bleu2, bleu3, bleu4))
            bleu1s.append(bleu1)
            bleu2s.append(bleu2)
            bleu3s.append(bleu3)
            bleu4s.append(bleu4)

        # 计算平均值
        bleu1_average = sum(bleu1s) / len(bleu1s)
        bleu2_average = sum(bleu2s) / len(bleu2s)
        bleu3_average = sum(bleu3s) / len(bleu3s)
        bleu4_average = sum(bleu4s) / len(bleu4s)

        # 输出
        print("average bleus: bleu1: %.3f, bleu2: %.3f, bleu4: %.3f" % (bleu1_average, bleu2_average, bleu4_average))
        return (bleu1_average, bleu2_average, bleu4_average)

    @staticmethod
    def em_score(references, candidates):
        total_cnt = len(references)
        match_cnt = 0
        for ref, cand in zip(references, candidates):
            if ref == cand:
                match_cnt = match_cnt + 1

        em_score = match_cnt / (float)(total_cnt)
        print("em_score: %.3f, match_cnt: %d, total_cnt: %d" % (em_score, match_cnt, total_cnt))
        return em_score

    @staticmethod
    def rouge_score(references, candidates):
        """
        rouge计算，NLG任务语句生成，词语的recall
        https://github.com/pltrdy/rouge
        :param references: list string
        :param candidates: list string
        :return:
        """
        rouge = Rouge()

        # 遍历计算rouge
        rouge1s = []
        rouge2s = []
        rougels = []
        for ref, cand in zip(references, candidates):
            ref = ' '.join(list(ref))
            cand = ' '.join(list(cand))
            rouge_score = rouge.get_scores(cand, ref)
            rouge_1 = rouge_score[0]["rouge-1"]['f']
            rouge_2 = rouge_score[0]["rouge-2"]['f']
            rouge_l = rouge_score[0]["rouge-l"]['f']
            # print "ref: %s, cand: %s" % (ref, cand)
            # print 'rouge_score: %s' % rouge_score

            rouge1s.append(rouge_1)
            rouge2s.append(rouge_2)
            rougels.append(rouge_l)

        # 计算平均值
        rouge1_average = sum(rouge1s) / len(rouge1s)
        rouge2_average = sum(rouge2s) / len(rouge2s)
        rougel_average = sum(rougels) / len(rougels)

        # 输出
        print("average rouges, rouge_1: %.3f, rouge_2: %.3f, rouge_l: %.3f" \
              % (rouge1_average, rouge2_average, rougel_average))
        return (rouge1_average, rouge2_average, rougel_average)


if __name__ == '__main__':
    references = ["腊八粥喝了吗", "我的机器人女友好好看啊", "那长沙明天天气呢"]
    candidates = ["腊八粥喝了吗", "机器人女友好好看啊", '长沙明天呢']

    # decode
    references = [ref for ref in references]
    candidates = [cand for cand in candidates]

    # 计算metrics
    Metrics.bleu_score(references, candidates)
    Metrics.em_score(references, candidates)
    Metrics.rouge_score(references, candidates)
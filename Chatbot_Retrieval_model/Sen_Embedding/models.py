import logging
import os
import time
import pathlib

import numpy as np
import tensorflow as tf
from onnxruntime import InferenceSession, SessionOptions

from Chatbot_Retrieval_model.transformers_bak.tokenization_bert import BertTokenizer


# import onnxmltools
# import onnxruntime


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SentenceBert:
    def __init__(self):
        pretrained_model_name = 'chinese-rbt3'
        embedding_onnx_model_path = 'saved_models/model_embedding.onnx'
        inference_onnx_model_path = 'saved_models/model_inference.onnx'
        # embedding_onnx_model_path = 'saved_models/model_embedding.onnx'
        # inference_onnx_model_path = 'saved_models/model_inference.onnx'
        embedding_onnx_options = SessionOptions()
        inference_onnx_options = SessionOptions()
        self.embedding_sess = InferenceSession(embedding_onnx_model_path, embedding_onnx_options,
                                               providers=["CUDAExecutionProvider"])
        self.inference_sess = InferenceSession(inference_onnx_model_path, inference_onnx_options,
                                               providers=["CUDAExecutionProvider"])
        self.max_length = 32
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.model_init()

    def model_init(self):
        # 这里初始化的sentence长度需要在2以上，因为self.tokenizer.batch_encode_plus
        sentence1_embedding = self.embedding(['恩', 'ok'])
        sentence2_embedding = self.embedding(['好的', '可以'])
        inference_inputs_onnx = {'input_1': sentence1_embedding, 'input_2': sentence2_embedding}
        self.inference_sess.run(None, inference_inputs_onnx)
        return

    def matching(self, sentence1, sentence2, topK=1):
        sentence1_num = len(sentence1)
        start_time = time.time()
        sentence1_embedding = self.embedding(sentence1)
        end_time = time.time()
        print(f'embedding cost time:{end_time - start_time}')
        sentence2_list = sentence2['sentenceList']
        try:
            start_time = time.time()
            sentence2_embedding = np.array(sentence2['sentenceEmbedding'], dtype=np.float32)
            end_time = time.time()
            print(f'array cost time:{end_time - start_time}')
        except:
            sentence2_embedding = self.embedding(sentence2_list)

        sentence2_num = len(sentence2_list)
        sentence1_embedding = np.tile(sentence1_embedding, [sentence2_num // sentence1_num, 1])
        inference_inputs_onnx = {'input_1': sentence1_embedding, 'input_2': sentence2_embedding}
        start_time = time.time()
        inference_pro = [x[0] for x in self.inference_sess.run(None, inference_inputs_onnx)[0].tolist()]
        end_time = time.time()
        print(f'inference cost time:{end_time - start_time}')
        inference_dict = dict(zip(sentence2_list, inference_pro))
        res = sorted(inference_dict.items(), key=lambda x: x[1], reverse=True)[:topK]
        return res

    def embedding(self, sentence):  # sentence type list
        sentence_token = self.tokenizer.batch_encode_plus(sentence, max_length=self.max_length,
                                                          return_tensors="tf", pad_to_max_length=True)['input_ids']
        sentence_num = sentence_token.shape[0]
        max_length_sentence = sentence_token.shape[1]
        sentence_padding = tf.concat([sentence_token, tf.zeros((sentence_num, self.max_length - max_length_sentence),
                                                               dtype=tf.int32)], axis=-1)
        embedding_inputs_onnx = {'input': sentence_padding.numpy()}
        sentence_embedding = self.embedding_sess.run(None, embedding_inputs_onnx)[0]
        return sentence_embedding


if __name__ == '__main__':
    sentencebert = SentenceBert()
    topK = 2
    sentence1 = ['好的']
    sentence2 = {"sentenceList": ["不行", "好的", "阿哈哈哈哈", "我爱你", "我叫湖心小笨酸", "啊啊啊", "卧槽你但也的", "我知道了我知道了", "我已经买过了", "原始数据、预处理后的训练数据"]}
    start_time = time.time()
    res = sentencebert.matching(sentence1, sentence2, topK)
    end_time = time.time()
    print(end_time-start_time)
    print(res)
    # sentenceList = [x.strip() for x in open('test/sentenceList', 'r', encoding='utf-8').readlines()]*5
    # sentenceEmbedding = sentencebert.embedding(sentenceList)
    # dict1 = {
    #     "sentence1": ["我现在在忙"],
    #     "sentence2": {
    #         "sentenceList": sentenceList,
    #         "sentenceEmbedding": sentenceEmbedding.tolist()
    #     },
    #     "topK": 10
    # }
    # json.dump(dict1, open('test/test3.json', 'w'), ensure_ascii=False)


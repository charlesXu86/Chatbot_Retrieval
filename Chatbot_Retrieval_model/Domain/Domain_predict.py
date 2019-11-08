# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   config.py

@Time    :   2019-10-29 17:35

@Desc    :   domain分类预测

'''

from Chatbot_Retrieval_model.Domain.domain_classifier import *
from Chatbot_Retrieval_model.Domain.config import Config
from tensorflow.python.estimator.model_fn import EstimatorSpec


cf = Config()

class Bert_Class():

    def __init__(self):
        self.graph_path = os.path.join(cf.pb_model_dir, 'classification_model.pb')
        self.ckpt_tool, self.pbTool = None, None
        self.prepare()

    def classification_model_fn(self, features, mode):
        with tf.gfile.GFile(self.graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_map = {"input_ids": input_ids, "input_mask": input_mask}
        pred_probs = tf.import_graph_def(graph_def, name='', input_map=input_map, return_elements=['pred_prob:0'])

        return EstimatorSpec(mode=mode, predictions={
            'encodes': tf.argmax(pred_probs[0], axis=-1),
            'score': tf.reduce_max(pred_probs[0], axis=-1)})

    def prepare(self):
        tokenization.validate_case_matches_checkpoint(cf.do_lower_case, cf.init_checkpoint)
        self.config = modeling.BertConfig.from_json_file(cf.bert_config_file)

        if cf.max_seq_length > self.config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (cf.max_seq_length, self.config.max_position_embeddings))

        # tf.gfile.MakeDirs(self.out_dir)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=cf.vocab_file,
                                                    do_lower_case=cf.do_lower_case)

        self.processor = DomainProcessor()
        self.train_examples = self.processor.get_train_examples(cf.data_dir)
        global label_list
        label_list = self.processor.get_labels()

        self.run_config = tf.estimator.RunConfig(
            model_dir=cf.output_dir, save_checkpoints_steps=cf.save_checkpoints_steps,
            tf_random_seed=None, save_summary_steps=100, session_config=None, keep_checkpoint_max=5,
            keep_checkpoint_every_n_hours=10000, log_step_count_steps=100, )

    def predict_on_ckpt(self, sentence):
        '''
        基于ckpt的模型预测
        :param sentence:
        :return:
        '''
        if not self.ckpt_tool:
            num_train_steps = int(len(self.train_examples) / cf.batch_size * cf.num_train_epochs)
            num_warmup_steps = int(num_train_steps * cf.warmup_proportion)

            model_fn = model_fn_builder(bert_config=self.config, num_labels=len(label_list),
                                        init_checkpoint=cf.init_checkpoint, learning_rate=cf.learning_rate,
                                        num_train=num_train_steps, num_warmup=num_warmup_steps, use_one_hot_embeddings=False)

            self.ckpt_tool = tf.estimator.Estimator(model_fn=model_fn, config=self.run_config, )
        exam = self.processor.one_example(sentence)  # 待预测的样本列表
        feature = convert_single_example(0, exam, label_list, cf.max_seq_length, self.tokenizer)

        predict_input_fn = input_fn_builder(features=[feature, ],
                                            seq_length=cf.max_seq_length, is_training=False,
                                            drop_remainder=False)
        result = self.ckpt_tool.predict(input_fn=predict_input_fn)  # 执行预测操作，得到一个生成器。
        gailv = list(result)[0]["probabilities"].tolist()
        pos = gailv.index(max(gailv))  # 定位到最大概率值索引，
        return label_list[pos]

    def predict_on_pb(self, sentence):
        '''
        基于pb格式的模型预测
        :param sentence:
        :return:
        '''
        if not self.pbTool:
            self.pbTool = tf.estimator.Estimator(model_fn=self.classification_model_fn, config=self.run_config, )
        exam = self.processor.one_example(sentence)  # 待预测的样本列表
        feature = convert_single_example(0, exam, label_list, cf.max_seq_length, self.tokenizer)
        predict_input_fn = input_fn_builder(features=[feature, ],
                                            seq_length=cf.max_seq_length, is_training=False,
                                            drop_remainder=False)
        result = self.pbTool.predict(input_fn=predict_input_fn)  # 执行预测操作，得到一个生成器。
        ele = list(result)[0]
        # print('类别：{}，置信度：{:.3f}'.format(label_list[ele['encodes']], ele['score']))
        return label_list[ele['encodes']]


# if __name__ == "__main__":
#     import time
#
#     testcase = ['我的车辆保险要交哪些']
#     toy = Bert_Class()
#     aaa = time.clock()
#     for t in testcase:
#         print(toy.predict_on_ckpt(t), t)
#     bbb = time.clock()
#     print('ckpt预测用时：', bbb - aaa)
#
#     aaa = time.clock()
#     for t in testcase:
#         toy.predict_on_pb(t)
#     bbb = time.clock()
#     print('pb模型预测用时：', bbb - aaa)

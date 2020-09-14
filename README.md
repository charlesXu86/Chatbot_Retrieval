# Chatbot_Retrieval
FAQ问答

## 项目目录结构
    .
    |____Chatbot_Retrieval                          # Django项目全局配置
    | |______init__.py
    | |____settings.py
    | |____urls.py
    | |____wsgi.py
    |____Chatbot_Retrival_rest                     # 项目的服务端                  
    | |____migrations
    | | |______init__.py
    | |______init__.py
    | |____Api
    | | |____QA
    | | | |____QA_server.py
    | | |____utils
    | | | |____LogUtils.py
    | | |____Domain
    | |____tests.py
    | |____urls.py
    | |____views.py
    |____README.md
    |____Chatbot_Retrieval_model                  # 项目的模型模块
    | |____util
    | | |____logutil.py
    | | |______init__.py
    | |______init__.py
    | |____QA                                     # 问答模型
    | | |____FAQ_v2.py
    | | |____data_preprocess.py
    | | |______init__.py
    | | |____dialogue_predict.py
    | | |____utils.py
    | | |____FAQ.py
    | |____Bert_sim                              # 相似度匹配模型
    | | |____.DS_Store
    | | |____config.py
    | | |______init__.py
    | | |____run_similarity.py
    | | |____test.py
    | |____Dialogue_utterance_rewriter           # 对话改写
    | | |____config.py
    | | |____util.py
    | | |____run_summarization.py
    | | |____inspect_checkpoint.py
    | | |____beam_search.py
    | | |____attention_decoder_softmax.py
    | | |____model.py
    | | |____model_metrics.py
    | | |____attention_decoder.py
    | | |____decode.py
    | | |____model_config.json
    | | |____batcher.py
    | | |____data.py
    | |____Domain                              # Domain分类
    | | |____config.py
    | | |____domain_classifier_v2.py
    | | |______init__.py
    | | |____domain_classifier.py
    | |____data_process
    |____templates
    |____manage.py
    |____data                                # 数据，部分数据未上传
    | |____FAQ
    | | |____FAQ.json
    | | |____FAQ.txt
    | | |____FAQ_baoxian
    | | |____FAQ_gouche
    | | |____baike_qa_car.json
    | | |____FAQ_xianliao
    | |____bert_sim
    | | |____dev.txt
    | | |____train.txt
    | | |____test.txt
    | | |____val.txt
    | |____dialogue_rewrite
    | | |____corpus.txt
    | |____domain
    | | |____dev.txt
    | | |____train.txt
    | | |____test.txt
    | | |____val.txt
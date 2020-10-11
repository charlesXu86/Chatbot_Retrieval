# Chatbot_Retrieval
FAQ问答


原理：
    
    1、基于es的文档检索（更改为自定义分词器）进行粗排序
    2、基于bert的语义匹配精排序。由于候选答案可能较多，使用onnx进行模型加速
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Convert pytorch checkpoints to TensorFlow 2.0 """

import argparse
import logging
import os

from xz_transformers import (
    BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    BertConfig,
    TFBertForPreTraining,
    TFBertForQuestionAnswering,
    TFBertForSequenceClassification,
    is_torch_available,
    load_pytorch_checkpoint_in_tf2_model,
    TF2_WEIGHTS_NAME
)

if is_torch_available():
    import torch
    import numpy as np
    from xz_transformers import (
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    )
else:
    (
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    ) = (
        None,
        None,
        None,
        None
    )

logging.basicConfig(level=logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

MODEL_CLASSES = {
    "bert": (
        BertConfig,
        TFBertForPreTraining,
        BertForPreTraining,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-large-uncased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-large-cased-whole-word-masking-finetuned-squad": (
        BertConfig,
        TFBertForQuestionAnswering,
        BertForQuestionAnswering,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ),
    "bert-base-cased-finetuned-mrpc": (
        BertConfig,
        TFBertForSequenceClassification,
        BertForSequenceClassification,
        BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
    )
}


def convert_pt_checkpoint_to_tf(model_type, pytorch_checkpoint_path, config_file, tf_dump_path,
                                compare_with_pt_model=False):
    """
    转换pt模型权重到tf2模型权重(.h5)
    :param model_type: pt模型类型，一般都是pretraining
    :param pytorch_checkpoint_path: pt模型路径xxx/pytorch_model.bin,或缩略名
    :param config_file: config文件路径,或缩略名
    :param tf_dump_path: tf2模型权重保存路径
    :param compare_with_pt_model: 是否与原pt模型进行比较
    :return:
    """
    if model_type not in MODEL_CLASSES:
        raise ValueError("Unrecognized model type, should be one of {}.".format(list(MODEL_CLASSES.keys())))

    config_class, model_class, pt_model_class, model_maps, config_map = MODEL_CLASSES[model_type]

    # Initialise TF model
    if config_file in config_map:
        config_file = config_map[config_file]
    config = config_class.from_json_file(config_file)
    config.output_hidden_states = True
    config.output_attentions = True
    print("Building TensorFlow model from configuration: {}".format(str(config)))
    tf_model = model_class(config)

    # Load weights from tf checkpoint
    if pytorch_checkpoint_path in model_maps:
        pytorch_checkpoint_path = model_maps[pytorch_checkpoint_path]

    # Load PyTorch checkpoint in tf2 model:
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path)

    if compare_with_pt_model:
        tfo = tf_model(tf_model.dummy_inputs, training=False)  # build the network

        state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
        pt_model = pt_model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )

        with torch.no_grad():
            pto = pt_model(**pt_model.dummy_inputs)

        np_pt = pto[0].numpy()
        np_tf = tfo[0].numpy()
        diff = np.amax(np.abs(np_pt - np_tf))
        print("Max absolute difference between models outputs {}".format(diff))
        assert diff <= 2e-2, "Error, model absolute difference is >2e-2: {}".format(diff)


    # Save pytorch-model
    print("Save TensorFlow model to {}".format(tf_dump_path))
    tf_model.save_weights(os.path.join(tf_dump_path, TF2_WEIGHTS_NAME), save_format="h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_dump_path", default="/Data/xen/Codes/xz_transformers/data/ner",
        type=str, help="Path to the output Tensorflow dump file."
    )
    parser.add_argument(
        "--model_type",
        default='bert',
        type=str,
        help="Model type selected in the list of {}. If not given, will download and convert all the models from AWS.".format(
            list(MODEL_CLASSES.keys())
        ),
    )
    parser.add_argument(
        "--pytorch_checkpoint_path",
        default='/Data/xen/Codes/xz_transformers/data/ner/bert-base-multilingual-cased-pytorch_model.bin',
        type=str,
        help="Path to the PyTorch checkpoint path or shortcut name to download from AWS. "
             "If not given, will download and convert all the checkpoints from AWS.",
    )
    parser.add_argument(
        "--config_file",
        default="/Data/xen/Codes/xz_transformers/data/ner/config.json",
        type=str,
        help="The config json file corresponding to the pre-trained model. \n"
             "This specifies the model architecture. If not given and "
             "--pytorch_checkpoint_path is not given or is a shortcut name"
             "use the configuration associated to the shortcut name on the AWS",
    )
    parser.add_argument(
        "--compare_with_pt_model", action="store_false", help="Compare Tensorflow and PyTorch model predictions."
    )
    args = parser.parse_args()
    convert_pt_checkpoint_to_tf(args.model_type.lower(),
                                args.pytorch_checkpoint_path,
                                args.config_file if args.config_file is not None else args.pytorch_checkpoint_path,
                                args.tf_dump_path,
                                compare_with_pt_model=args.compare_with_pt_model)

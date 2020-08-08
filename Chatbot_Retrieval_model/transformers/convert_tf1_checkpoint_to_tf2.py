# -*- coding: utf-8 -*-
# Created by xieenning at 2020/2/5
import logging
import os
import torch
import numpy
import re
import tensorflow as tf
from xz_transformers import BertConfig, BertForPreTraining, TFBertForPreTraining

logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# step 1
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    xz_name_tuple = {}
    for name, array in zip(names, arrays):
        original_name = name
        model_attr_names = []
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            # layer_0
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                # ['layer', '0', '']
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
                model_attr_names.append("weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
                model_attr_names.append("bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
                model_attr_names.append("weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
                model_attr_names.append("classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                    model_attr_names.append(str(scope_names[0]))
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
                # TODO
                model_attr_names.append(str(num))
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
            model_attr_names.append("weight")
        elif m_name == "kernel":
            # TODO
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
        xz_name_tuple[original_name] = '.'.join(model_attr_names)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for tmp_key, tmp_value in xz_name_tuple.items():
        print(f"{tmp_key} --> {tmp_value}")
    return model


def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove=""):
    """ Convert a TF 2.0 model variable name in a pytorch model weight name.

        Conventions for TF2.0 scopes -> PyTorch attribute names conversions:
            - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
            - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

        return tuple with:
            - pytorch model weight name
            - transpose: boolean indicating weither TF2.0 and PyTorch weights matrices are transposed with regards to each other
    """
    tf_name = tf_name.replace(":0", "")  # device ids
    tf_name = re.sub(
        r"/[^/]*___([^/]*)/", r"/\1/", tf_name
    )  # '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
    tf_name = tf_name.replace(
        "_._", "/"
    )  # '_._' is replaced by a level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)
    tf_name = re.sub(r"//+", "/", tf_name)  # Remove empty levels at the end
    tf_name = tf_name.split("/")  # Convert from TF2.0 '/' separators to PyTorch '.' separators
    tf_name = tf_name[1:]  # Remove level zero

    # When should we transpose the weights
    transpose = bool(tf_name[-1] == "kernel" or "emb_projs" in tf_name or "out_projs" in tf_name)

    # Convert standard TF2.0 names in PyTorch names
    if tf_name[-1] == "kernel" or tf_name[-1] == "embeddings" or tf_name[-1] == "gamma":
        tf_name[-1] = "weight"
    if tf_name[-1] == "beta":
        tf_name[-1] = "bias"

    # Remove prefix if needed
    tf_name = ".".join(tf_name)
    if start_prefix_to_remove:
        tf_name = tf_name.replace(start_prefix_to_remove, "", 1)

    return tf_name, transpose


def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False):
    """ Load pytorch state_dict in a TF 2.0 model.
    """
    try:
        import torch  # noqa: F401
        import tensorflow as tf  # noqa: F401
        from tensorflow.python.keras import backend as K
    except ImportError:
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    if tf_inputs is None:
        tf_inputs = tf_model.dummy_inputs

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure model is built

    # Adapt state dict - TODO remove this and update the AWS weights files instead
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in pt_state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        pt_state_dict[new_key] = pt_state_dict.pop(old_key)

    # Make sure we are able to load PyTorch base models as well as derived models (with heads)
    # TF models always have a prefix, some of PyTorch models (base ones) don't
    start_prefix_to_remove = ""
    if not any(s.startswith(tf_model.base_model_prefix) for s in pt_state_dict.keys()):
        # "bert."
        start_prefix_to_remove = tf_model.base_model_prefix + "."

    symbolic_weights = tf_model.trainable_weights + tf_model.non_trainable_weights
    tf_loaded_numel = 0
    weight_value_tuples = []
    all_pytorch_weights = set(list(pt_state_dict.keys()))
    xz_name_dict_ = {}
    for symbolic_weight in symbolic_weights:
        sw_name = symbolic_weight.name
        name, transpose = convert_tf_weight_name_to_pt_weight_name(
            sw_name, start_prefix_to_remove=start_prefix_to_remove
        )


        # Find associated numpy array in pytorch model state dict
        if name not in pt_state_dict:
            if allow_missing_keys:
                continue
            raise AttributeError("{} not found in PyTorch model".format(name))

        array = pt_state_dict[name].numpy()

        # TODO
        if transpose:
            array = numpy.transpose(array)

        if len(symbolic_weight.shape) < len(array.shape):
            array = numpy.squeeze(array)
        elif len(symbolic_weight.shape) > len(array.shape):
            array = numpy.expand_dims(array, axis=0)

        try:
            assert list(symbolic_weight.shape) == list(array.shape)
        except AssertionError as e:
            e.args += (symbolic_weight.shape, array.shape)
            raise e

        tf_loaded_numel += array.size
        # logger.warning("Initialize TF weight {}".format(symbolic_weight.name))

        weight_value_tuples.append((symbolic_weight, array))
        all_pytorch_weights.discard(name)
        xz_name_dict_[name] = sw_name

    K.batch_set_value(weight_value_tuples)

    print('-------------------------------------')
    for tmp_key, tmp_value in xz_name_dict_.items():
        print(f'{tmp_key} --> {tmp_value}')

    if tf_inputs is not None:
        tf_model(tf_inputs, training=False)  # Make sure restore ops are run

    logger.info("Loaded {:,} parameters in the TF 2.0 model.".format(tf_loaded_numel))

    logger.info("Weights or buffers not loaded from PyTorch model: {}".format(all_pytorch_weights))

    return tf_model


# step 2
def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """ Load pytorch checkpoints in a TF 2.0 model
    """
    try:
        import tensorflow as tf  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        logger.error(
            "Loading a PyTorch model in TensorFlow, requires both PyTorch and TensorFlow to be installed. Please see "
            "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    pt_path = os.path.abspath(pytorch_checkpoint_path)
    logger.info("Loading PyTorch weights from {}".format(pt_path))

    pt_state_dict = torch.load(pt_path, map_location="cpu")
    # pt_state_dict_keys = pt_state_dict.keys()
    logger.info("PyTorch checkpoint contains {:,} parameters".format(sum(t.numel() for t in pt_state_dict.values())))

    return load_pytorch_weights_in_tf2_model(
        tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys
    )


def run():
    BASE_PATH = '/Data/xen/Codes/transformers_xz/pretrained_models/tensorflow1.x/chinese_roberta_tiny_zhuiyi'
    tf_checkpoint_path = os.path.join(BASE_PATH, 'bert_model.ckpt')
    pytorch_dump_path = os.path.join(BASE_PATH, 'pytorch_model.bin')
    tf_dump_path = os.path.join(BASE_PATH, 'tf_model.h5')

    config = BertConfig.from_json_file(os.path.join(BASE_PATH, 'bert_config.json'))
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)
    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)

    config = BertConfig.from_json_file(os.path.join(BASE_PATH, 'bert_config.json'))
    config.output_hidden_states = True
    config.output_attentions = True
    print("Building TensorFlow model from configuration: {}".format(str(config)))
    tf_model = TFBertForPreTraining(config)

    # Load PyTorch checkpoint in tf2 model:
    tf_model = load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_dump_path)

    # Save pytorch-model
    print("Save TensorFlow model to {}".format(tf_dump_path))
    tf_model.save_weights(tf_dump_path, save_format="h5")


if __name__ == '__main__':
    # BASE_PATH = '/Data/public/Bert/chinese_wwm_ext_L-12_H-768_A-12'
    # tf_checkpoint_path = os.path.join(BASE_PATH, 'bert_model.ckpt')
    # init_vars = tf.train.list_variables(tf_checkpoint_path)  # type: list
    #
    # for weight_index, (weight_name, weight_shape) in enumerate(init_vars):
    #     print(f'{weight_index}: {weight_name}, {weight_shape}')
    #
    # config = BertConfig.from_json_file(os.path.join(BASE_PATH, 'bert_config.json'))
    # print("Building PyTorch model from configuration: {}".format(str(config)))
    # model = BertForPreTraining(config)
    #
    # load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    run()


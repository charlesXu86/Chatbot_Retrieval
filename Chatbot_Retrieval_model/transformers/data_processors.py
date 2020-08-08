# -*- coding: utf-8 -*-
# Created by xieenning at 2020/3/11
""" data processors and helpers """

import logging
import os
import pickle
import pandas as pd

from .file_utils import is_tf_available
from .data_utils import DataProcessor, InputExample, InputFeatures, InputExampleLabeling

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        return_tensors=None,
        save_id2label_path=None
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the tasks.
        tokenizer: Instance of a tokenizer that will tokenize the tasks
        max_length: Maximum tasks length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the tasks will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
        return_tensors
        save_id2label_path

    Returns:
        If the ``tasks`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    label_map = {label: i for i, label in enumerate(label_list)}
    if save_id2label_path:
        id2label = {tmp_value: tmp_key for tmp_key, tmp_value in label_map.items()}
        tmp_dir = os.path.dirname(save_id2label_path)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        with open(save_id2label_path, 'wb') as f:
            pickle.dump(id2label, f, -1)
        logger.info(f"Saved label map to '{save_id2label_path}'.")
    len_examples = len(examples)
    all_inputs = []
    batch_length = -1
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing tasks %d/%d" % (ex_index, len_examples))
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, max_length=max_length)
        input_ids = inputs["input_ids"]
        all_inputs.append(inputs)
        if len(input_ids) > batch_length:
            batch_length = len(input_ids)

    # padding part
    features = []
    for (ex_index, (tmp_inputs, example)) in enumerate(zip(all_inputs, examples)):
        input_ids = tmp_inputs["input_ids"]
        token_type_ids = tmp_inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = tmp_inputs["attention_mask"]
        # Zero-pad up to the sequence length.
        padding_length = batch_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        assert len(input_ids) == batch_length, "Error with input length {} vs {}".format(
            len(input_ids), batch_length
        )
        assert len(attention_mask) == batch_length, "Error with input length {} vs {}".format(
            len(attention_mask), batch_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise ValueError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    if return_tensors is None:
        return features
    elif return_tensors == "tf":
        if not is_tf_available():
            raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        dataset = tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )
        return dataset


def convert_examples_to_features_labeling(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=30,  # 30 o
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        return_tensors=None,
        save_id2label_path=None
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}
    if save_id2label_path:
        id2label = {tmp_value: tmp_key for tmp_key, tmp_value in label_map.items()}
        tmp_dir = os.path.dirname(save_id2label_path)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        with open(save_id2label_path, 'wb') as f:
            pickle.dump(id2label, f, -1)
        logger.info(f"Saved label map to '{save_id2label_path}'.")

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_added_tokens()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label=label_ids)
        )
    if return_tensors == "tf":
        if not is_tf_available():
            raise RuntimeError("return_tensors set to 'tf' but TensorFlow 2.0 can't be imported")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        dataset = tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int32),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([None]),
            ),
        )
        return dataset
    return features


class SequencePairClassificationProcessor(DataProcessor):
    """Processor for the Sequence Pair Classification data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"),\
               self._create_examples2(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"),\
               self._create_examples2(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"),\
               self._create_examples2(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates tasks for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]  # sentence1
            text_b = line[1]  # sentence2
            label = line[2]  # label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples2(self, lines, set_type):
        """Creates tasks for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]  # sentence2
            text_b = line[0]  # sentence1
            label = line[2]  # label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SingleSentenceClassificationProcessor(DataProcessor):
    """ Generic processor for a single sentence classification data set."""

    def __init__(self):
        self.labels = None

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy())
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir=None):
        """See base class."""
        if data_dir is None:
            return self.labels
        else:
            return sorted(pd.read_csv(os.path.join(data_dir, "train.tsv"), header=0, sep='\t',
                                      dtype={'label': str}).label.unique())

    def _create_examples(self, lines, set_type):
        """Creates tasks for the training and dev sets."""
        examples = []
        added_labels = set()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line[0]  # sentence
            label = line[1]  # label
            added_labels.add(label)
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        if set_type == 'train':
            self.labels = sorted(list(added_labels))
        return examples


class SequenceLabelingProcessor(DataProcessor):
    """ Generic processor for sentence labeling data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["words"].numpy().decode("utf-8"),
            str(tensor_dict["labels"].numpy())
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.txt")))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, data_dir=None):
        """See base class."""
        if data_dir:
            with open(data_dir, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def _create_examples(self, lines, set_type):
        """Creates tasks for the training and dev sets."""
        examples = []
        guid_index = 1
        words = []
        labels = []
        for line in lines:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExampleLabeling(guid="{}-{}".format(set_type, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExampleLabeling(guid="{}-{}".format(set_type, guid_index), words=words, labels=labels))
        return examples


class SentenceBertEmbeddingProcessor(DataProcessor):
    """ Generic processor for a single sentence classification data set."""

    def __init__(self):
        self.labels = None

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy())
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train1.tsv")))
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train2.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train1.tsv")), "train"),\
               self._create_examples(self._read_tsv(os.path.join(data_dir, "train2.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev1.tsv")), "dev"),\
               self._create_examples(self._read_tsv(os.path.join(data_dir, "dev2.tsv")), "dev")


    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test1.tsv")), "test"),\
               self._create_examples(self._read_tsv(os.path.join(data_dir, "test2.tsv")), "test")

    def get_labels(self, data_dir=None):
        """See base class."""
        if data_dir is None:
            return self.labels
        else:
            return sorted(pd.read_csv(os.path.join(data_dir, "train1.tsv"), header=0, sep='\t',
                                      dtype={'label': str}).label.unique())

    def _create_examples(self, lines, set_type):
        """Creates tasks for the training and dev sets."""
        examples = []
        added_labels = set()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line[0]  # sentence
            label = line[1]  # label
            added_labels.add(label)
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        if set_type == 'train':
            self.labels = sorted(list(added_labels))
        return examples



processors = {
    "spc": SequencePairClassificationProcessor,
    "ssc": SingleSentenceClassificationProcessor,
    "sl": SequenceLabelingProcessor
}

output_modes = {
    "spc": "classification",
    "ssc": "classification",
    "sl": "classification"
}

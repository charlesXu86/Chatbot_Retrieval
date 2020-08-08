# -*- coding: utf-8 -*-

'''
@Author  :   Xu

@Software:   PyCharm

@File    :   model.py

@Time    :   2019-11-06 14:25

@Desc    :  1、adam 0.001，好像grad 不clip结果不错

            2、This is the top-level file to train, evaluate or test your summarization model

'''

import sys
import time
import os
import tensorflow as tf
import numpy as np
import json
from collections import namedtuple
from Chatbot_Retrieval_model.Dialogue_utterance_rewriter.data import Vocab
from Chatbot_Retrieval_model.Dialogue_utterance_rewriter.batcher import Batcher
from Chatbot_Retrieval_model.Dialogue_utterance_rewriter.model import SummarizationModel
from Chatbot_Retrieval_model.Dialogue_utterance_rewriter.decode import BeamSearchDecoder
from Chatbot_Retrieval_model.Dialogue_utterance_rewriter import util
from Chatbot_Retrieval_model.Dialogue_utterance_rewriter.config import Config
from tensorflow.python import debug as tf_debug


cf = Config()
def calc_running_avg_loss(loss,
                          running_avg_loss,
                          summary_writer,
                          step,
                          decay=0.99):
    """Calculate the running average loss via exponential decay.
  This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

  Args:
    loss: loss on the most recent eval step
    running_avg_loss: running_avg_loss so far
    summary_writer: FileWriter object to write for tensorboard
    step: training iteration step
    decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

  Returns:
    running_avg_loss: new running average loss
  """
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


def restore_best_model():
    """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
    tf.logging.info("Restoring bestmodel for training...")

    # Initialize all vars in the model
    sess = tf.Session(config=util.get_config())
    print("Initializing all variables...")
    sess.run(tf.global_variables_initializer())

    # Restore the best model from eval dir
    saver = tf.train.Saver(
        [v for v in tf.all_variables() if "Adagrad" not in v.name])
    print("Restoring all non-adagrad variables from best model in eval dir...")
    curr_ckpt = util.load_ckpt(saver, sess, "eval")
    print("Restored %s." % curr_ckpt)

    # Save this model to train dir and quit
    new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
    new_fname = os.path.join(cf.log_root, "train", new_model_name)
    print("Saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver(
    )  # this saver saves all variables that now exist, including Adagrad variables
    new_saver.save(sess, new_fname)
    print("Saved.")
    exit()


def convert_to_coverage_model():
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=util.get_config())
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([
        v for v in tf.global_variables()
        if "coverage" not in v.name and "Adagrad" not in v.name
    ])
    print("restoring non-coverage variables...")
    curr_ckpt = util.load_ckpt(saver, sess)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver(
    )  # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def setup_training(model, batcher):
    """Does setup before starting training (run_training)"""
    train_dir = os.path.join(cf.log_root, "train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    model.build_graph()
    if cf.convert_to_coverage_model:
        assert cf.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
        convert_to_coverage_model()
    if cf.restore_best_model:
        restore_best_model()
    saver = tf.compat.v1.train.Saver(max_to_keep=3)  # keep 3 checkpoints at a time

    sv = tf.train.Supervisor(
        logdir=train_dir,
        is_chief=True,
        saver=saver,
        summary_op=None,
        save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
        save_model_secs=60,  # checkpoint every 60 secs
        global_step=model.global_step)
    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(
        config=util.get_config())
    tf.logging.info("Created session.")
    try:
        run_training(
            model, batcher, sess_context_manager, sv,
            summary_writer)  # this is an infinite loop until interrupted
    except KeyboardInterrupt:
        tf.logging.info(
            "Caught keyboard interrupt on worker. Stopping supervisor...")
        sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
    tf.logging.info("starting run_training")
    with sess_context_manager as sess:
        if cf.debug:  # start the tensorflow debugger
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        while True:  # repeats until interrupted
            batch = batcher.next_batch()

            tf.logging.info('running training step...')
            t0 = time.time()
            results = model.run_train_step(sess, batch)
            t1 = time.time()
            tf.logging.info('seconds for training step: %.3f', t1 - t0)

            loss = results['loss']
            tf.logging.info('loss: %f', loss)  # print the loss to screen

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            if cf.coverage:
                coverage_loss = results['coverage_loss']
                tf.logging.info(
                    "coverage_loss: %f",
                    coverage_loss)  # print the coverage loss to screen

            # get the summaries and iteration number so we can write summaries to tensorboard
            summaries = results[
                'summaries']  # we will write these summaries to tensorboard using summary_writer
            train_step = results[
                'global_step']  # we need this to update our running average loss

            summary_writer.add_summary(summaries,
                                       train_step)  # write the summaries
            if train_step % 100 == 0:  # flush the summary writer every so often
                summary_writer.flush()


def run_eval(model, batcher, vocab):
    """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
    model.build_graph()
    saver = tf.train.Saver(max_to_keep=3)
    sess = tf.Session(config=util.get_config())
    # make a subdir of the root dir for eval data
    eval_dir = os.path.join(cf.log_root, "eval")
    # this is where checkpoints of best models are saved
    bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')
    summary_writer = tf.summary.FileWriter(eval_dir)
    running_avg_loss = 0  # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
    best_loss = None  # will hold the best loss achieved so far

    while True:
        _ = util.load_ckpt(saver, sess)  # load a new checkpoint
        batch = batcher.next_batch()  # get the next batch

        # run eval on the batch
        t0 = time.time()
        results = model.run_eval_step(sess, batch)
        t1 = time.time()
        tf.logging.info('seconds for batch: %.2f', t1 - t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        tf.logging.info('loss: %f', loss)
        if cf.coverage:
            coverage_loss = results['coverage_loss']
            tf.logging.info("coverage_loss: %f", coverage_loss)

        # add summaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        # calculate running avg loss
        running_avg_loss = calc_running_avg_loss(
            np.asscalar(loss), running_avg_loss, summary_writer, train_step)

        # If running_avg_loss is best so far, save this checkpoint (early stopping).
        # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
        if best_loss is None or running_avg_loss < best_loss:
            tf.logging.info(
                'Found new best model with %.3f running_avg_loss. Saving to %s',
                running_avg_loss, bestmodel_save_path)
            saver.save(
                sess,
                bestmodel_save_path,
                global_step=train_step,
                latest_filename='checkpoint_best')
            best_loss = running_avg_loss

        # flush the summary writer every so often
        if train_step % 100 == 0:
            summary_writer.flush()


def main():
    # if len(unused_argv) != 1:  # prints a message if you've entered cf incorrectly
    #     raise Exception("Problem with cf: %s" % unused_argv)

    tf.logging.set_verbosity(
        tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting seq2seq_attention in %s mode...', (cf.mode))

    # Change log_root to cf.log_root/cf.exp_name and create the dir if necessary
    cf.log_root = os.path.join(cf.log_root, cf.exp_name)
    if not os.path.exists(cf.log_root):
        if cf.mode == "train":
            os.makedirs(cf.log_root)
        else:
            raise Exception(
                "Logdir %s doesn't exist. Run in train mode to create it." %
                (cf.log_root))

    vocab = Vocab(cf.vocab_file, cf.vocab_size)  # create a vocabulary

    # If in decode mode, set batch_size = beam_size
    # Reason: in decode mode, we decode one example at a time.
    # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
    if cf.mode == 'decode':
        cf.batch_size = cf.beam_size

    # If single_pass=True, check we're in decode mode
    if cf.single_pass and cf.mode != 'decode':
        raise Exception(
            "The single_pass flag should only be True in decode mode")

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = [
        'mode', 'learning_rate', 'adagrad_init_acc', 'rand_unif_init_mag',
        'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim',
        'batch_size', 'encoder_type', 'max_dec_steps', 'max_enc_steps', 'coverage',
        'cov_loss_wt', 'pointer_gen'
    ]

    hps_dict = {}

    json_file = open('model_config.json', 'r', encoding='utf-8')
    param = json.load(json_file)
    for key, val in param.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(
        cf.data_dir, vocab, cf, single_pass=cf.single_pass)

    tf.set_random_seed(42)  # a seed value for randomness

    if cf.mode == 'train':
        print("creating model...")
        model = SummarizationModel(cf, vocab)
        setup_training(model, batcher)
    elif cf.mode == 'eval':
        model = SummarizationModel(cf, vocab)
        run_eval(model, batcher, vocab)
    elif cf.mode == 'decode':
        decode_model_hps = cf  # This will be the hyperparameters for the decoder model

        # The model is configured with max_dec_steps=1
        # because we only ever run one step of the decoder at a time (to do beam search).
        # Note that the batcher is initialized with max_dec_steps equal to e.g. 100
        # because the batches need to contain the full summaries
        decode_model_hps = cf._replace(max_dec_steps=1)
        model = SummarizationModel(decode_model_hps, vocab)
        decoder = BeamSearchDecoder(model, batcher, vocab)
        # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
        decoder.decode()
    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")


if __name__ == '__main__':
    main()

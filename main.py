# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import tensorflow as tf
from config import Config
from data_loader import DataLoader
from model import PointerNetworks

def do_train(args):
    tf.logging.info("Training rnn model")
    args.mode = "train"
    config = Config(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    tf.logging.info('loading')
    data_loader = DataLoader(config)
    data_loader.read_data(args.name)

    tf.logging.info('building')
    # train_data, train_labels, train_mask
    with tf.Graph().as_default():
        tf.logging.info("Building model...", )
        start = time.time()
        model = PointerNetworks(config)
        tf.logging.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=tf_config) as session:
            session.run(init)
            model.fit(session, saver, data_loader)

def do_evaluate(args):
    tf.logging.info("Evaluating rnn model")
    args.mode = "test"
    config = Config(args)

    print(" -- loading -- ")
    data_loader = DataLoader("processed")
    data_loader.read_data(args.name)

    with tf.Graph().as_default():
        tf.logging.info("Building model...", )
        start = time.time()
        model = PointerNetworks(config)

        tf.logging.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        with tf.Session(config=tf_config) as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            tf.logging.info("Evaluating test set")
                # preds = model.evaluate_test(session, data_loader)

def main():
    parser = argparse.ArgumentParser(description='Trains and tests an Seq2Seq model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-gpu', '--gpu_id', default="1", help="GPU id.")
    command_parser.add_argument('-fp', '--file_path', default="tsp5", help="File path.")
    command_parser.add_argument('-is_train', '--is_train', default=True, type=bool, help="Training mode.")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    command_parser.add_argument('-m', '--model-path', help="Training data")
    command_parser.add_argument('-gpu', '--gpu_id', default="1", help="GPU id.")
    command_parser.add_argument('-fp', '--file_path', default="tsp5", help="File path.")
    command_parser.add_argument('-is_train', '--is_train', default=True, type=bool, help="Training mode.")
    command_parser.set_defaults(func=do_evaluate)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)

if __name__ == "__main__":
    main()
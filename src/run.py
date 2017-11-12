from __future__ import print_function, division

import numpy as np
import tensorflow as tf

import sys, os, time, json
from easydict import EasyDict as edict

from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags

# required arguments
flags.DEFINE_string('model', None, 'The name of the model. model_dir should contain configs/ and checkpoint/ directories')
flags.DEFINE_string('config', 'default', 'Config for the model')

# optional arguments
flags.DEFINE_boolean('train', True, 'True for training, False for testing phase. Default [True]')
flags.DEFINE_string('test_set', 'train', 'Dataset used for testing. Default: ["train"]')
# flags.DEFINE_boolean('test', True, 'True for training, False for testing phase. Default [True]')
flags.DEFINE_string('gpu', None, 'GPU number. Default [None]')
FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    model_dir = os.path.join(Models, FLAGS.model)
    sys.path.append(model_dir)

    # load in model specific configurations
    config_dir = os.path.join(model_dir, FLAGS.config)
    config_path = os.path.join(config_dir, 'config.json')
    with open(config_path, 'r+') as f:
        config = edict(json.load(f))

    # check for existing checkpoints
    train_dir = os.path.join(config_dir, 'train')
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt:
        print('Latest checkpoint:', ckpt.model_checkpoint_path)
    elif not FLAGS.train:
        print('Cannot find checkpoint to test from, exiting')
        exit()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    # load model with the given configurations
    from load_model import load_model
    model = load_model(config, sess)

    # dataset specific loading
    sys.path.append(os.path.join(Data, config.dataset))
    from load_dataset import load_dataset, save_predictions

    # create saver and load in data
    if FLAGS.train:
        print('Training')
        saver = tf.train.Saver(model.get_train_variables())
        train_data = load_dataset('train') # tuple of (videos, poses)
        val_data = load_dataset('val')
    else:
        print('Testing')
        saver = tf.train.Saver(model.get_test_variables())
        test_data = load_dataset('test')

    # initialize model
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        model.init_pretrained_weights()

    if FLAGS.train:
        # train model
        summary_writer = tf.summary.FileWriter(train_dir)
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        success = model.train(saver, summary_writer, train_data, val_data, checkpoint_path)
        if not success:
            exit()
        test_data = load_dataset(FLAGS.test_set)
        print('Testing')

    # test model
    test_output_dir = os.path.join(config_dir, FLAGS.test_set, 'outputs')
    test_predictions = model.test(test_data)
    save_predictions(test_predictions, test_output_dir)

if __name__ == '__main__':
    tf.app.run()

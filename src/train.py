import numpy as np
import tensorflow as tf

import os, time, json
from easydict import EasyDict as edict

flags = tf.app.flags

# required arguments
flags.DEFINE_string('model_dir', None, 'The directory of the model. model_dir should contain configs/ and checkpoint/ directories')
flags.DEFINE_string('config_name', None, 'The json config file name (without the .json). Checkpoint will be named after config name')

# optional arguments
flags.DEFINE_boolean('train', False, 'True for training, False for testing phase. Default [False]')
flags.DEFINE_string('gpu', None, 'GPU number. Default [None]')
FLAGS = flags.FLAGS

def main(argv):
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    # load in model specific configurations
    config_dir = os.path.join(FLAGS.model_dir, 'configs')
    config_path = os.path.join(config_dir, FLAGS.config_name)
    with open(config_path, 'r+') as f:
        config = edict(json.load(f))

    # check for existing checkpoints
    train_dir = os.path.join(config_dir, 'train')
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if not ckpt and not FLAGS.train:
        print('Cannot find checkpoint to test from, exiting')
        exit()

    # load model with the given configurations
    sys.path.append(FLAGS.model_dir)
    from load_model import load_model
    model = load_model(config, sess)

    # dataset specific loading
    sys.path.append(Data, config.dataset)
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
        sess.run(tf.global_variable_initializer())
        model.init_pretrained_weights()

    if FLAGS.train:
        # train model
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        model.train(saver, summary_writer, train_data, val_data, checkpoint_path)
        test_data = load_dataset('test')
        print('Testing')

    # test model
    test_predictions = model.test(test_data)
    save_predictions(test_predictions)

if __name__ == '__main__':
    tf.app.run()

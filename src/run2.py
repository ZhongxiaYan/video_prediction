from __future__ import print_function, division

import numpy as np
import tensorflow as tf

import sys, os, time, json
from easydict import EasyDict as edict

from util import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags

# optional arguments
flags.DEFINE_boolean('train', True, 'True for training, False for testing phase. Default [True]')
flags.DEFINE_string('gpu', '0', 'GPU number. Default [0]')
flags.DEFINE_string('overrides', '', 'Override for parameters in the config file. Specify like "--overrides lr=0.5,l2_loss=0.1". Also need to specify new_config')
flags.DEFINE_string('new_config', None, 'New config directory to make to store new json with overrides')
flags.DEFINE_string('save_root', '/media/deoraid03/jeff/video_prediction/', 'Checkpoints will be saved in subdirectories of this root. Symlinks will point to train subdirectories. Default: ["/media/deoraid03/jeff/video_prediction/"]')
FLAGS = flags.FLAGS

def apply_overrides(config, override_string):
    print('Applying overrides')
    for name, val in (s.split('=') for s in override_string.split(',')):
        if name not in config:
            raise RuntimeError('Existing config does not contain %s' % name)
        print(name, '=', val)
        config[name] = type(config[name])(val)
        
def main(argv):
    assert len(argv) == 2, 'Only argument must be path to config directory'
    config_dir = os.path.abspath(argv[1])
    assert config_dir.startswith(Models), 'Invalid config directory %s' % config_dir
    model_name, config_name = config_dir[len(Models):].split('/')[:2]

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    model_dir = os.path.join(Models, model_name)
    sys.path.append(model_dir)

    # load in model specific configurations
    config_dir = os.path.join(model_dir, config_name)
    config_path = os.path.join(config_dir, 'config.json')
    with open(config_path, 'r+') as f:
        config = edict(json.load(f))
        
    if FLAGS.overrides:
        assert FLAGS.new_config, 'Must set FLAGS.new_config to new config name'
        config_name = FLAGS.new_config
        config_dir = os.path.join(model_dir, FLAGS.new_config)
        config_path = os.path.join(config_dir, 'config.json')
        apply_overrides(config, FLAGS.overrides)
        make_dir(config_dir)
        with open(config_path, 'w+') as f:
            json.dump(config, f, indent=4)
        print('Created new config %s at %s' % (FLAGS.new_config, config_dir))
    
    # check for existing checkpoints
    train_dir = os.path.join(config_dir, 'train')
    if FLAGS.save_root:
        for dname in ['train', 'val', 'test']:
            orig_dir = os.path.join(config_dir, dname)
            save_dir = os.path.join(FLAGS.save_root, 'models', model_name, config_name, dname)
            make_dir(save_dir)

            if not os.path.exists(orig_dir):
                os.symlink(save_dir, orig_dir)
                print('Creating symlink %s -> %s' % (orig_dir, save_dir))
            elif not os.path.islink(orig_dir):
                raise RuntimeError('%s exists but is not a link. Cannot create new link to %s' % (orig_dir, save_dir))
        
    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt:
        print('Latest checkpoint:', ckpt.model_checkpoint_path)
    elif not FLAGS.train:
        raise RuntimeError('Cannot find checkpoint to test from, exiting')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    # dataset specific loading
    sys.path.append(os.path.join(Data, config.dataset))
    from load_dataset import load_dataset
    dataset = load_dataset(config)
    
    # load model with the given configurations
    from load_model import load_model
    model = load_model(config, sess, dataset)

    # create saver and load in data 
    if FLAGS.train:
        print('Training')
        saver = tf.train.Saver(model.get_train_variables(), max_to_keep=None)
    else:
        print('Testing')
        saver = tf.train.Saver(model.get_test_variables())

    # initialize model
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        model.init_pretrained_weights(sess)

    train_output_dir = os.path.join(config_dir, 'train', 'output')
    val_output_dir = os.path.join(config_dir, 'val', 'output')
    test_output_dir = os.path.join(config_dir, 'test', 'output')
    if FLAGS.train:
        # train model
        summary_writer = tf.summary.FileWriter(train_dir)
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        success = model.train(saver, summary_writer, checkpoint_path, train_output_dir, val_output_dir)
        if not success:
            print('Error... Exiting script')
            exit()
        print('Testing')

    # test model
    test_predictions = model.test(train_output_dir, test_output_dir)

if __name__ == '__main__':
    tf.app.run()

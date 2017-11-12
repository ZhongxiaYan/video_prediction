from __future__ import print_function, division

import numpy as np
import tensorflow as tf

from util import *

class NNBase(object):
    def __init__(self, config, sess):
        self.config = config
        self.sess = sess
        self.layer_outputs = {}

    def train(self, saver, summary_writer, train_data, val_data, train_dir):
        pass

    def test(self, test_data):
        pass
        
    def get_train_variables(self):
        return tf.global_variables()

    def get_test_variables(self):
        return tf.global_variables()

    def vgg_shallow(self, input):
        scope = tf.get_variable_scope().name
        pool_ = lambda x: max_pool(x, 2, 2)
        conv_ = lambda x, output_depth, name: conv(x, 3, output_depth, 1, name=name)
        
        prev = input
        for i, depth in enumerate([64, 128, 256, 512, 512]):
            name = 'conv%s_1' % (i + 1)
            convi_1 = conv_(prev, depth, name)
            self.layer_outputs[scope + '/' + name] = convi_1
            prev = pool_(convi_1)
        
        flattened = tf.contrib.layers.flatten(prev)
        fc_6 = dropout(fc(flattened, 4096, 'fc6'), 0.5)
        fc_7 = fc(fc_6, 4096, 'fc7', relu=False)
        return fc_7

    def C1(self, input, reuse=False):
        alex_input = tf.image.resize_images(input, [227, 227])
        with tf.variable_scope('C1', reuse=reuse):
            return alexnet(alex_input)

    def load_alexnet_weights(self, scopes):
        weights_dict = np.load(Models + 'alexnet.npy', encoding='bytes').item()
        for scope in scopes:
            with tf.variable_scope(scope, reuse=True):
                for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                    with tf.variable_scope(layer):
                        W_value, b_value = weights_dict[layer]
                        W = tf.get_variable('W', trainable=False)
                        b = tf.get_variable('b', trainable=False)
                        self.sess.run(W.assign(W_value))
                        self.sess.run(b.assign(b_value))

    def load_vgg_weights(self, scopes):
        weights_dict = np.load(Models + 'vgg16.npy', encoding='bytes').item()
        weights_dict = { key.decode('ascii') : value for key, value in weights_dict.items() }
        for scope in scopes:
            with tf.variable_scope(scope, reuse=True):
                for layer in ['conv1_1', 'conv1_2',
                              'conv2_1', 'conv2_2',
                              'conv3_1', 'conv3_2', 'conv3_3',
                              'conv4_1', 'conv4_2', 'conv4_3',
                              'conv5_1', 'conv5_2', 'conv5_3',
                              'fc6', 'fc7']:
                    with tf.variable_scope(layer):
                        W_value, b_value = weights_dict[layer]
                        W = tf.get_variable('W')
                        b = tf.get_variable('b')
                        self.sess.run(W.assign(W_value))
                        self.sess.run(b.assign(b_value))

import numpy as np
import tensorflow as tf

import sys
sys.path.append(Src)

from util import *
from base_model import NNBase

def load_model(config, sess):
    return VanillaGAN(config, sess)

class VanillaGAN(NNBase):
    def __init__(self, config, sess):
        super().__init__(config, sess)
        self.global_step = tf.get_variable('global_step', initializer=0, trainable=False)

        self.f_t = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.p_t = tf.placeholder(tf.float32, [None, 224, 224, 1])
        self.f_t_n = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.p_t_n = tf.placeholder(tf.float32, [None, 224, 224, 1])

        # generator output
        f_t_n_pred = self.generator_network(self.f_t, self.p_t, self.p_t_n)

        # discriminator outputs
        pr_real_gen = self.discriminator_network(f_t_n_pred, self.p_t_n)
        pr_real_true = self.discriminator_network(self.f_t_n, self.p_t_n, reuse=True)
        pr_real_mismatch = self.discriminator_network(self.f_t, self.p_t_n, reuse=True)

        # generator losses
        l2_loss_gen = l2_loss(self.f_t_n, f_t_n_pred)
        C1_f_t = self.C1(self.f_t)
        feat_loss_gen = l2_loss(C1_f_t, self.C1(f_t_n_pred, reuse=True))
        adv_loss_gen = -tf.reduce_mean(tf.log(pr_real_gen))
        self.loss_gen = config.l2_coef * l2_loss_gen + config.feat_coef * feat_loss_gen + config.adv_coef * adv_loss_gen

        # discriminator losses
        self.loss_disc = -tf.reduce_mean(tf.log(pr_real_true) + 0.5 * tf.log(1 - pr_real_gen) + 0.5 * tf.log(1 - pr_real_mismatch))
        
        variables_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.opt_gen = tf.train.GradientDescentOptimizer(learning_rate=config.lr_gen).minimize(self.loss_gen, var_list=variables_gen, global_step=self.global_step)

        variables_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.opt_disc = tf.train.GradientDescentOptimizer(learning_rate=config.lr_disc).minimize(self.loss_disc, var_list=variables_disc)

    def train(self, saver, summary_writer, train_data, val_data, checkpoint_path):
        sess = self.sess
        config = self.config
        train_videos, train_poses = train_data

        disp_loss_gen = disp_loss_disc = 0
        while True:
            frames1, frames2, poses1, poses2 = get_minibatch(config.batch_size, train_videos, train_poses)
            disp_loss_gen += self.generator.fit_batch(frames1, poses1, frames2, poses2) / config.disp_interval
            disp_loss_disc += self.discriminator.fit_batch(frames1, poses1, frames2, poses2) / config.disp_interval
        
            g_step = tf.train.global_step(sess, self.global_step)

            if g_step % config.disp_interval == 0:
                print('train step %s: gen loss=%.4f, disc loss=%.4f' % (g_step, disp_loss_gen, disp_loss_disc))
                disp_loss_gen = 0
                disp_loss_disc = 0
            # if g_step % config.val_interval == 0:
            #     loss_gen = 0
            #     loss_disc = 0
            #     for _ in range(config.val_batches):
            #         loss_gen += self.generator.predict() / config.val_batches
            #         loss_disc += self.discriminator.predict() / config.val_batches
            #     print('val step %s: gen loss=%.4f, disc loss=%.4f' % (g_step, loss_gen, loss_disc))
            if g_step % config.save_interval == 0:
                saver.save(train_dir, checkpoint_path, global_step=g_step)
            if g_step >= self.max_steps:
                break

    def test(self, test_data):
        test_videos, test_poses = test_data
        frames1, poses1, frames2, poses2 = get_minibatch(config.test_size, test_videos, test_poses)
        predicted2 = self.generator.predict(frames1, poses1, frames2, poses2)
        return zip(frames1, poses1, frames2, poses2, predicted2)

    def generator_network(self, f_t, p_t, p_t_n):
        with tf.variable_scope('generator'):
            p_t_n_latent = self.f_pose(p_t_n)
            latent = p_t_n_latent - self.f_pose(p_t, force_reuse=True) + self.f_img(f_t)
            return self.f_dec(latent)

    def discriminator_network(self, f, p, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            f_latent = f_img(f)
            p_latent = f_pose(p)
            concat = tf.concat([f_latent, p_latent], axis=1)
            fc8 = fc(concat, 1024, name='fc8')
            output = tf.nn.sigmoid(fc(fc8, 1, name='fc9', relu=False))
        return output
    
    def f_pose(self, input, reuse=False):
        with tf.variable_scope('f_pose', reuse=reuse):
            return vgg_simple(input)
        
    def f_img(self, input):
        with tf.variable_scope('f_img'):
            return vgg(input, process_input=True)
        
    def f_dec(self, input):
        with tf.variable_scope('f_dec'):
            reshaped = tf.reshape(input, shape=[tf.shape(input)[0], 1, 1, 4096])
            return vgg_deconv(reshaped)
        
    def C1(self, input, reuse=False):
        alex_input = tf.image.resize_images(input, [227, 227])
        with tf.variable_scope('C1', reuse=reuse):
            return alexnet(alex_input)
    
    def init_pretrained_weights(self):
        pretrained_models = self.config.pretrained_models
        weights_dict = np.load(pretrained_models['alexnet'], encoding='bytes').item()
        with tf.variable_scope('C1', reuse=True):
            for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
                with tf.variable_scope(layer):
                    W_value, b_value = weights_dict[layer]
                    W = tf.get_variable('W', trainable=False)
                    b = tf.get_variable('b', trainable=False)
                    self.sess.run(W.assign(W_value))
                    self.sess.run(b.assign(b_value))
        weights_dict = np.load(pretrained_models['vgg'], encoding='bytes').item()
        weights_dict = { key.decode('ascii') : value for key, value in weights_dict.items() }
        for scope in 'generator/f_img', 'discriminator/f_img':
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

    def fit_batch_gen(self, f_t, p_t, f_t_n, p_t_n):
        _, loss = self.sess.run((self.opt, self.loss), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss

    def fit_batch_disc(self, f_t, p_t, f_t_n, p_t_n):
        _, loss = self.sess.run((self.opt, self.loss), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss
    
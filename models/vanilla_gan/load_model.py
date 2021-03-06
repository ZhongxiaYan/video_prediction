import numpy as np
import tensorflow as tf

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
        self.f_t_n_pred = self.generator_network(self.f_t, self.p_t, self.p_t_n)

        # discriminator outputs
        pr_real_gen = self.discriminator_network(self.f_t_n_pred, self.p_t_n)
        pr_real_true = self.discriminator_network(self.f_t_n, self.p_t_n, reuse=True)
        pr_real_mismatch = self.discriminator_network(self.f_t, self.p_t_n, reuse=True)

        with tf.variable_scope('generator'):
            # generator losses
            l2_loss_gen = l2_loss(self.f_t_n, self.f_t_n_pred)
            C1_f_t = self.C1(self.f_t)
            feat_loss_gen = l2_loss(C1_f_t, self.C1(self.f_t_n_pred, reuse=True))
            adv_loss_gen = -tf.reduce_mean(tf.log(pr_real_gen))
            self.loss_gen = config.l2_coef * l2_loss_gen + config.adv_coef * adv_loss_gen + config.feat_coef * feat_loss_gen
            self.summs_gen = tf.summary.merge([
                tf.summary.scalar('l2_loss', l2_loss_gen),
                tf.summary.scalar('feat_loss', feat_loss_gen),
                tf.summary.scalar('adv_loss', adv_loss_gen),
                tf.summary.scalar('loss', self.loss_gen)
            ])

        with tf.variable_scope('discriminator'):
            # discriminator losses
            real_loss = -tf.reduce_mean(tf.log(pr_real_true))
            gen_loss = -tf.reduce_mean(tf.log(1 - pr_real_gen))
            mismatch_loss = -tf.reduce_mean(tf.log(1 - pr_real_mismatch))
            self.loss_disc = config.real_coef * real_loss + config.gen_coef * gen_loss + config.mismatch_coef * mismatch_loss
            self.summs_disc = tf.summary.merge([
                tf.summary.scalar('real_loss', real_loss),
                tf.summary.scalar('gen_loss', gen_loss),
                tf.summary.scalar('mismatch_loss', mismatch_loss),
                tf.summary.scalar('loss', self.loss_disc)
            ])

        variables_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.opt_gen = tf.train.GradientDescentOptimizer(learning_rate=config.lr_gen).minimize(self.loss_gen, var_list=variables_gen, global_step=self.global_step)

        variables_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        self.opt_disc = tf.train.GradientDescentOptimizer(learning_rate=config.lr_disc).minimize(self.loss_disc, var_list=variables_disc)

    def train(self, saver, summary_writer, train_data, val_data, checkpoint_path):
        sess = self.sess
        config = self.config
        train_videos, train_poses = train_data

        disp_loss_gen = disp_loss_disc = 0
        g_step = tf.train.global_step(sess, self.global_step)
        while g_step < config.max_steps:
            frames1, poses1, frames2, poses2 = get_minibatch(config.batch_size, train_videos, train_poses)
            loss_gen, summs_gen = self.fit_batch_gen(frames1, poses1, frames2, poses2)
            loss_disc, summs_disc = self.fit_batch_disc(frames1, poses1, frames2, poses2)

            disp_loss_gen += loss_gen / config.disp_interval
            disp_loss_disc += loss_disc / config.disp_interval
        
            g_step = tf.train.global_step(sess, self.global_step)
            if g_step % config.summary_interval == 0:
                summary_writer.add_summary(summs_gen, g_step)
                summary_writer.add_summary(summs_disc, g_step)
            if g_step % config.disp_interval == 0:
                print('train step %s: gen loss=%.4f, disc loss=%.4f' % (g_step, disp_loss_gen, disp_loss_disc))
                if np.isnan(disp_loss_gen) or np.isnan(disp_loss_disc):
                    return False
                disp_loss_gen = disp_loss_disc = 0
            if g_step % config.save_interval == 0:
                saver.save(sess, checkpoint_path, global_step=g_step)
        return True

    def test(self, test_data):
        test_videos, test_poses = test_data
        frames1, poses1, frames2, poses2 = get_minibatch(self.config.test_size, test_videos, test_poses)
        predicted2 = self.predict(frames1, poses1, poses2)
        return zip(frames1, poses1, frames2, poses2, predicted2)

    def generator_network(self, f_t, p_t, p_t_n):
        with tf.variable_scope('generator'):
            p_t_n_latent = self.f_pose(p_t_n)
            latent = p_t_n_latent - self.f_pose(p_t, reuse=True) + self.f_img(f_t)
            return self.f_dec(latent)

    def discriminator_network(self, f, p, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            f_latent = self.f_img(f)
            p_latent = self.f_pose(p)
            concat = tf.concat([f_latent, p_latent], axis=1)
            fc8 = fc(concat, 1024, name='fc8')
            return tf.nn.sigmoid(fc(fc8, 1, name='fc9', relu=False))
    
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
        self.load_alexnet_weights(['generator/C1'])
        self.load_vgg_weights(['generator/f_img', 'discriminator/f_img'])

    def fit_batch_gen(self, f_t, p_t, f_t_n, p_t_n):
        _, loss, summs = self.sess.run((self.opt_gen, self.loss_gen, self.summs_gen), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss, summs

    def fit_batch_disc(self, f_t, p_t, f_t_n, p_t_n):
        _, loss, summs = self.sess.run((self.opt_disc, self.loss_disc, self.summs_disc), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss, summs
    
    def predict(self, f_t, p_t, p_t_n):
        return self.sess.run(self.f_t_n_pred, feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t })

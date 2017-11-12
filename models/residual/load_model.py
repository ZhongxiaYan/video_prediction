import numpy as np
import tensorflow as tf

from util import *
from base_model import NNBase

def load_model(config, sess):
    return ResidualGAN(config, sess)

class ResidualGAN(NNBase):
    def __init__(self, config, sess):
        super(ResidualGAN, self).__init__(config, sess)
        self.global_step = tf.get_variable('global_step', initializer=0, trainable=False)

        self.f_t = tf.placeholder(tf.float32, [None, 224, 224, 1])
        self.p_t = tf.placeholder(tf.float32, [None, 224, 224, 1])
        self.f_t_n = tf.placeholder(tf.float32, [None, 224, 224, 1])
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
            adv_loss_gen = -tf.reduce_mean(tf.log(pr_real_gen))

            pose_map = (self.p_t + self.p_t_n) / 256.0
            pose_loss_gen = l2_loss(self.f_t_n * pose_map, self.f_t_n_pred * pose_map)

            self.loss_gen = config.l2_coef * l2_loss_gen \
                          + config.adv_coef * adv_loss_gen \
                          + config.pose_coef * pose_loss_gen
            self.summs_gen = tf.summary.merge([
                tf.summary.scalar('l2_loss', l2_loss_gen),
                tf.summary.scalar('adv_loss', adv_loss_gen),
                tf.summary.scalar('pose_loss', pose_loss_gen),
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
        self.opt_gen = tf.train.AdamOptimizer(learning_rate=config.lr_gen).minimize(self.loss_gen, var_list=variables_gen, global_step=self.global_step)

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
            return self.vgg_shallow(input)
        
    def f_dec(self, input):
        l_outs = self.layer_outputs
        with tf.variable_scope('f_dec'):
            reshaped = tf.reshape(input, shape=[tf.shape(input)[0], 1, 1, 4096])

            deconv6_2 = deconv(reshaped, 7, 128, 1, 'deconv6_2', padding='VALID')
            deconv6_1 = deconv(deconv6_2, 3, 128, 2, 'deconv6_1')

            deconv5_2 = deconv(deconv6_1, 3, 128, 1, 'deconv5_2')
            deconv5_1 = deconv(deconv5_2, 3, 128, 2, 'deconv5_1')
            
            deconv4_2 = deconv(deconv5_1, 3, 128, 1, 'deconv4_2')
            deconv4_1 = deconv(deconv4_2, 3, 64, 2, 'deconv4_1')
            deconv4 = tf.concat([deconv4_1, l_outs['generator/f_img/conv3_1']], axis=-1)

            deconv3_2 = deconv(deconv4, 3, 64, 1, 'deconv3_2')
            deconv3_1 = deconv(deconv3_2, 3, 32, 2, 'deconv3_1')

            deconv2_2 = deconv(deconv3_1, 3, 32, 1, 'deconv2_2')
            deconv2_1 = deconv(deconv2_2, 3, 16, 2, 'deconv2_1')
            deconv2 = tf.concat([deconv2_1, l_outs['generator/f_img/conv1_1']], axis=-1)

            deconv1_2 = deconv(deconv2, 3, 16, 1, 'deconv1_2')
            deconv1_1 = deconv(deconv1_2, 3, 1, 1, 'deconv1_1')
            return deconv1_1
    
    def init_pretrained_weights(self):
        pass

    def fit_batch_gen(self, f_t, p_t, f_t_n, p_t_n):
        _, loss, summs = self.sess.run((self.opt_gen, self.loss_gen, self.summs_gen), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss, summs

    def fit_batch_disc(self, f_t, p_t, f_t_n, p_t_n):
        _, loss, summs = self.sess.run((self.opt_disc, self.loss_disc, self.summs_disc), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss, summs
    
    def predict(self, f_t, p_t, p_t_n):
        return self.sess.run(self.f_t_n_pred, feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t })

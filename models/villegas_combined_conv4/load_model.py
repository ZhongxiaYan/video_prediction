import numpy as np
import tensorflow as tf

from util import *
from base_model import NNBase

def load_model(config, sess, dataset):
    return Network(config, sess, dataset)

class Network(NNBase):
    def __init__(self, config, sess, dataset):
        super(Network, self).__init__(config, sess)
        self.global_step = tf.get_variable('global_step', initializer=0, trainable=False)
        self.dataset = dataset
        
        config.input_depth = 1 if config.gray else 3
        
        self.f_t = tf.placeholder(tf.float32, [None, 224, 224, config.input_depth])
        self.p_t = tf.placeholder(tf.float32, [None, 224, 224, config.L])
        self.f_t_n = tf.placeholder(tf.float32, [None, 224, 224, config.input_depth])
        self.p_t_n = tf.placeholder(tf.float32, [None, 224, 224, config.L])
        self.f_t_norm = (self.f_t / 255. - 0.5) * 2
        self.p_t_norm = (self.p_t / 255. - 0.5) * 2
        self.f_t_n_norm = (self.f_t_n / 255. - 0.5) * 2
        self.p_t_n_norm = (self.p_t_n / 255. - 0.5) * 2

        # generator output
        self.f_t_n_pred = self.generator_network(self.f_t_norm, self.p_t_norm, self.p_t_n_norm)

        # discriminator outputs
        pr_real_gen = self.discriminator_network(self.f_t_n_pred, self.p_t_n_norm)
        pr_real_true = self.discriminator_network(self.f_t_n_norm, self.p_t_n_norm, reuse=True)
        pr_real_mismatch = self.discriminator_network(self.f_t_norm, self.p_t_n_norm, reuse=True)

        with tf.variable_scope('generator'):
            # generator losses
            l2_loss_gen = l2_loss(self.f_t_n_norm, self.f_t_n_pred)
            adv_loss_gen = -tf.reduce_mean(tf.log(pr_real_gen))

            self.loss_gen = config.l2_coef * l2_loss_gen \
                          + config.adv_coef * adv_loss_gen
            self.summs_gen = tf.summary.merge([
                tf.summary.scalar('l2_loss', l2_loss_gen),
                tf.summary.scalar('adv_loss', adv_loss_gen),
                tf.summary.scalar('loss', self.loss_gen)
            ])

        with tf.variable_scope('discriminator'):
            # discriminator losses
            real_loss = -tf.reduce_mean(tf.log(tf.maximum(pr_real_true, 0.01)))
            gen_loss = -tf.reduce_mean(tf.log(1.01 - pr_real_gen))
            mismatch_loss = -tf.reduce_mean(tf.log(1.01 - pr_real_mismatch))
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

    def train(self, saver, summary_writer, checkpoint_path, train_output_dir, val_output_dir):
        sess = self.sess
        config = self.config
        dataset = self.dataset

        disp_loss_gen = disp_loss_disc = 0
        g_step = tf.train.global_step(sess, self.global_step)
        while g_step < config.max_steps:
            frames1, poses1, frames2, poses2 = dataset.get_train_batch(config.batch_size)
            loss_gen, summs_gen = self.fit_batch_gen(frames1, poses1, frames2, poses2)
            loss_disc, summs_disc = self.fit_batch_disc(frames1, poses1, frames2, poses2)
            
            if np.isnan(loss_gen) or np.isnan(loss_disc):
                print('gen loss=%s, disc loss=%s' % (loss_gen, loss_disc))
                return False
            disp_loss_gen += loss_gen / config.disp_interval
            disp_loss_disc += loss_disc / config.disp_interval
        
            g_step = tf.train.global_step(sess, self.global_step)
            if g_step % config.summary_interval == 0:
                summary_writer.add_summary(summs_gen, g_step)
                summary_writer.add_summary(summs_disc, g_step)
            if g_step % config.disp_interval == 0:
                print('train step %s: gen loss=%.4f, disc loss=%.4f' % (g_step, disp_loss_gen, disp_loss_disc))
                disp_loss_gen = disp_loss_disc = 0
            if g_step % config.save_interval == 0:
                print('train step %s: generating samples' % g_step)
                for output_base, test in zip([train_output_dir, val_output_dir], [False, True]):
                    output_dir = output_base + '-' + str(g_step) + '/'
                    for action, vnames, (frames1, poses1, frames2, poses2) in dataset.get_val_batch(test=test):
                        predicted = np.round(self.predict(frames1, poses1, poses2)).astype(np.uint8)
                        action_dir = output_dir + action + '/'
                        make_dir(action_dir)
                        for args in zip(vnames, frames1, poses1, frames2, poses2, predicted):
                            dataset.save_image_predictions(output_dir, action, *args)
                print('train step %s: saving model' % g_step)
                saver.save(sess, checkpoint_path, global_step=g_step)
        return True

    def test(self, train_output_dir, test_output_dir):
        config = self.config
        g_step = tf.train.global_step(self.sess, self.global_step)
        for output_base, test in zip([train_output_dir, test_output_dir], [False, True]):
            output_dir = output_base + '-' + str(g_step) + '/'
            for action, vnames, videos, pose_list in self.dataset.get_test_batch(test=test):
                action_dir = output_dir + action + '/'
                make_dir(action_dir)
                for vname, video, poses in zip(vnames, videos, pose_list):
                    f0, p0 = video[0:1], poses[0:1]
                    predicted = [self.predict(f0, p0, poses[i : i + 1]) for i in range(1, len(video))]
                    predicted = np.round(np.concatenate([f0] + predicted, axis=0)).astype(np.uint8)
                    self.dataset.save_video_predictions(output_dir, action, vname, predicted)

    def generator_network(self, f_t, p_t, p_t_n):
        with tf.variable_scope('generator'):
            p_t_n_latent = self.f_pose(p_t_n)
            latent = self.f_img(f_t) + p_t_n_latent - self.f_pose(p_t, reuse=True)
            return self.f_dec(latent)

    def discriminator_network(self, f, p, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            concat = tf.concat([f, p], axis=-1)
            conv6 = self.vgg_pool3(concat)
            flattened = tf.contrib.layers.flatten(conv6)
            fc7 = dropout(fc(flattened, 1024, name='fc7'), 0.5)
            return tf.nn.sigmoid(fc(fc7, 1, name='fc8', relu=False))
    
    def f_pose(self, input, reuse=False):
        with tf.variable_scope('f_pose', reuse=reuse):
            return self.vgg_pool3(input)
        
    def f_img(self, input):
        with tf.variable_scope('f_img'):
            return self.vgg_pool3(input)
        
    def vgg_pool3(self, input):
        scope = tf.get_variable_scope().name
        pool_ = lambda x: max_pool(x, 2, 2)
        conv_ = lambda x, output_depth, name: conv(x, 3, output_depth, 1, name=name)
        
        if 'width' in self.config:
            width = self.config.width
        else:    
            width = 1
        depths = [int(64*width), int(128*width), int(256*width), 512]
        prev = input
        for i, depth in enumerate(depths):
            layer_num = i + 1
            name = 'conv%s_1' % layer_num
            convi_1 = conv_(prev, depth, name)
            name = 'conv%s_2' % layer_num
            convi_2 = conv_(convi_1, depth, name)
            if layer_num == 3 or layer_num == 4:
                name = 'conv%s_3' % layer_num
                convi_3 = conv_(convi_2, depth, name)
                prev = pool_(convi_3)
            else:    
                prev = pool_(convi_2)
        return prev
    
    def f_dec(self, input):
        l_outs = self.layer_outputs
        if 'width' in self.config:
            width = self.config.width
        else:    
            width = 1
        with tf.variable_scope('f_dec'):
            deconv4_4 = deconv(input, 3, int(512*width), 2, 'deconv4_4')            
            deconv4_3 = deconv(deconv4_4, 3, int(512*width), 1, 'deconv4_3')
            deconv4_2 = deconv(deconv4_3, 3, int(512*width), 1, 'deconv4_2')
            deconv4_1 = deconv(deconv4_2, 3, int(256*width), 2, 'deconv4_1')
                      
            deconv3_3 = deconv(deconv4_1, 3, int(256*width), 1, 'deconv3_3')
            deconv3_2 = deconv(deconv3_3, 3, int(256*width), 1, 'deconv3_2')
            deconv3_1 = deconv(deconv3_2, 3, int(128*width), 2, 'deconv3_1')
            
            deconv2_2 = deconv(deconv3_1, 3, int(128*width), 1, 'deconv2_2')
            deconv2_1 = deconv(deconv2_2, 3, int(64*width), 2, 'deconv2_1')

            deconv1_2 = deconv(deconv2_1, 3, int(64*width), 1, 'deconv1_2')
            deconv1_1 = deconv(deconv1_2, 3, self.config.input_depth, 1, 'deconv1_1', tanh=True)
            return deconv1_1
    
    def init_pretrained_weights(self, sess):
        if 'pretrained' in self.config:
            weights_dict = np.load('../models/vgg16.npy', encoding='bytes').item()
            weights_dict = { key.decode('ascii') : value for key, value in weights_dict.items() }
            with tf.variable_scope('generator/f_img', reuse=True):
                for layer in ['conv1_2',
                              'conv2_1', 'conv2_2',
                              'conv3_1', 'conv3_2', 'conv3_3',
                              ]:
                    with tf.variable_scope(layer):
                        W_value, b_value = weights_dict[layer]
                        W = tf.get_variable('W')
                        b = tf.get_variable('b')
                        sess.run(W.assign(W_value))
                        sess.run(b.assign(b_value))
        

    def fit_batch_gen(self, f_t, p_t, f_t_n, p_t_n):
        _, loss, summs = self.sess.run((self.opt_gen, self.loss_gen, self.summs_gen), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss, summs

    def fit_batch_disc(self, f_t, p_t, f_t_n, p_t_n):
        _, loss, summs = self.sess.run((self.opt_disc, self.loss_disc, self.summs_disc), feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t, self.f_t_n : f_t_n })
        return loss, summs
    
    def predict(self, f_t, p_t, p_t_n):
        return (self.sess.run(self.f_t_n_pred, feed_dict={ self.p_t : p_t, self.p_t_n : p_t_n, self.f_t : f_t }) / 2. + 0.5) * 255.

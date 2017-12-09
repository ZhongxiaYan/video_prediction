import numpy as np
import tensorflow as tf

from util import *
from base_model import NNBase

def load_model(config, sess, dataset):
    return Network(config, sess, dataset)

def normalize(x):
    return (x / 255. - 0.5) * 2

def denormalize(x):
    return (x / 2. + 0.5) * 255.

def get_feed_dict(params, args):
    return { p : a for p, a in zip(params, args) }

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
        
        f_t_norm = normalize(self.f_t)
        p_t_norm = normalize(self.p_t)
        f_t_n_norm = normalize(self.f_t_n)
        p_t_n_norm = normalize(self.p_t_n)

        # generator output
        f_t_n_pred_norm = self.generator_network(self.f_t, p_t_norm, self.f_t_n, p_t_n_norm)
        self.f_t_n_pred = denormalize(f_t_n_pred_norm)

        # discriminator outputs
        pr_real_gen = self.discriminator_network(f_t_n_pred_norm, p_t_n_norm)
        pr_real_true = self.discriminator_network(f_t_n_norm, p_t_n_norm, reuse=True)
        pr_real_mismatch = self.discriminator_network(f_t_norm, p_t_n_norm, reuse=True)

        with tf.variable_scope('generator'):
            # generator losses
            l2_loss_gen = l2_loss(self.f_t_n, self.f_t_n_pred)
            adv_loss_gen = -tf.reduce_mean(tf.log(tf.maximum(pr_real_gen, 0.01)))
            self.loss_gen = config.l2_coef * l2_loss_gen + config.adv_coef * adv_loss_gen
            summ_list = [
                tf.summary.scalar('l2_loss', l2_loss_gen),
                tf.summary.scalar('adv_loss', adv_loss_gen)
            ]
            if config.latent_coef:
                latent_loss_gen = l2_loss(self.layer_outputs['f_t_n_latent_true'], self.layer_outputs['f_t_n_latent_pred']) / tf.reduce_mean(self.layer_outputs['f_t_n_latent_true'] ** 2)
                self.loss_gen += config.latent_coef * latent_loss_gen
                summ_list.append(tf.summary.scalar('latent_loss', latent_loss_gen))
            summ_list.append(tf.summary.scalar('loss', self.loss_gen))
            self.summs_gen = tf.summary.merge(summ_list)

        with tf.variable_scope('discriminator'):
            # discriminator losses
            real_loss = -tf.reduce_mean(tf.log(tf.maximum(pr_real_true, 0.01)))
            gen_loss = -tf.reduce_mean(tf.log(tf.maximum(1 - pr_real_gen, 0.01)))
            mismatch_loss = -tf.reduce_mean(tf.log(tf.maximum(1 - pr_real_mismatch, 0.01)))
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
        self.opt_disc = tf.train.AdamOptimizer(learning_rate=config.lr_disc).minimize(self.loss_disc, var_list=variables_disc)

    def train(self, saver, summary_writer, checkpoint_path, train_output_dir, val_output_dir):
        sess = self.sess
        config = self.config
        dataset = self.dataset

        disp_loss_gen = disp_loss_disc = 0
        g_step = tf.train.global_step(sess, self.global_step)
        train_params = [self.f_t, self.p_t, self.f_t_n, self.p_t_n]
        test_params = [self.f_t, self.p_t, self.p_t_n]
        while g_step < config.max_steps:
            train_batch = dataset.get_train_batch(config.batch_size)
            feed_dict = get_feed_dict(train_params, train_batch)
            loss_gen, summs_gen = self.fit_batch_gen(feed_dict)
            loss_disc, summs_disc = self.fit_batch_disc(feed_dict)
            
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
                print('train step %s: saving model' % g_step)
                saver.save(sess, checkpoint_path, global_step=g_step)
                print('train step %s: generating samples' % g_step)
                for output_base, test in zip([train_output_dir, val_output_dir], [False, True]):
                    output_dir = output_base + '-' + str(g_step) + '/'
                    for action, vnames, (frames1, poses1, frames2, poses2) in dataset.get_val_batch(test=test):
                        feed_dict = get_feed_dict(test_params, [frames1, poses1, poses2])
                        predicted = np.round(self.predict(feed_dict)).astype(np.uint8)
                        action_dir = output_dir + action + '/'
                        make_dir(action_dir)
                        for args in zip(vnames, frames1, poses1, frames2, poses2, predicted):
                            dataset.save_image_predictions(output_dir, action, *args)
        return True

    def test(self, train_output_dir, test_output_dir):
        config = self.config
        g_step = tf.train.global_step(self.sess, self.global_step)
        params = [self.f_t, self.p_t, self.p_t_n]
        for output_base, test in zip([train_output_dir, test_output_dir], [False, True]):
            output_dir = output_base + '-' + str(g_step) + '/'
            for action, vnames, videos, pose_list in self.dataset.get_test_batch(test=test):
                action_dir = output_dir + action + '/'
                make_dir(action_dir)
                for vname, video, poses in zip(vnames, videos, pose_list):
                    f0, p0 = video[0:1], poses[0:1]
                    predicted = [self.predict(get_feed_dict(params, [f0, p0, poses[i : i + 1]])) for i in range(1, len(video))]
                    predicted = np.round(np.concatenate([f0] + predicted, axis=0)).astype(np.uint8)
                    self.dataset.save_video_predictions(output_dir, action, vname, predicted)

    def generator_network(self, f_t, p_t, f_t_n, p_t_n):
        with tf.variable_scope('generator'):
            latent = self.f_img(f_t) + self.f_pose(p_t_n) - self.f_pose(p_t, reuse=True)
            self.layer_outputs['f_t_n_latent_pred'] = latent
            f_img_t_n = self.f_img(f_t_n, reuse=True)
            self.layer_outputs['f_t_n_latent_true'] = f_img_t_n
            return self.f_dec(latent)

    def discriminator_network(self, f, p, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            concat = tf.concat([f, p], axis=-1)
            conv6 = self.vgg_conv(concat, standard=False)
            flattened = tf.contrib.layers.flatten(conv6)
            fc7 = dropout(fc(flattened, 1024, name='fc7'), 0.5)
            return tf.nn.sigmoid(fc(fc7, 1, name='fc8', relu=False))
    
    def f_pose(self, input, reuse=False):
        with tf.variable_scope('f_pose', reuse=reuse):
            return self.vgg_conv(input, standard=False)
        
    def f_img(self, input, reuse=False):
        with tf.variable_scope('f_img', reuse=reuse):
            return self.vgg_conv(input, standard=True)
        
    def vgg_conv(self, input, standard=False):
        fix_weight_layers = 0 if not standard else self.config.fix_weight_layers
        if standard:
            VGG_MEAN = [103.939, 116.779, 123.68]

            # Convert RGB to BGR and subtract mean
            red, green, blue = tf.split(input, 3, axis=3)
            input = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], axis=3)
            
            def conv_fixed_relu(x, num_filters, name):
                input_channels = int(x.get_shape()[-1])
                with tf.variable_scope(name):
                    W = tf.get_variable('W', shape=[3, 3, input_channels, num_filters], trainable=False)
                    b = tf.get_variable('b', shape=[num_filters], trainable=False)
                    output = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
                return tf.nn.relu(output)

        pool_ = lambda x: max_pool(x, 2, 2)
        conv_lrelu = lambda x, output_depth, name: conv(x, 3, output_depth, 1, name=name)

        depths = [64, 128, 256, 512]
        num_sublayers = [2, 2, 3, 3]
        prev = input
        for i, depth in enumerate(depths):
            layer_num = i + 1
            if i < fix_weight_layers:
                conv_ = conv_fixed_relu
            else:
                conv_ = conv_lrelu
            for sublayer_num in range(1, num_sublayers[i] + 1):
                name = 'conv%s_%s' % (layer_num, sublayer_num)
                prev = conv_(prev, depth, name)
            prev = pool_(prev)
        return prev

    def f_dec(self, input):
        with tf.variable_scope('f_dec'):
            deconv4_4 = deconv(input, 3, 512, 2, 'deconv4_4')            
            deconv4_3 = deconv(deconv4_4, 3, 512, 1, 'deconv4_3')
            deconv4_2 = deconv(deconv4_3, 3, 512, 1, 'deconv4_2')
            deconv4_1 = deconv(deconv4_2, 3, 256, 2, 'deconv4_1')
                      
            deconv3_3 = deconv(deconv4_1, 3, 256, 1, 'deconv3_3')
            deconv3_2 = deconv(deconv3_3, 3, 256, 1, 'deconv3_2')
            deconv3_1 = deconv(deconv3_2, 3, 128, 2, 'deconv3_1')
            
            deconv2_2 = deconv(deconv3_1, 3, 128, 1, 'deconv2_2')
            deconv2_1 = deconv(deconv2_2, 3, 64, 2, 'deconv2_1')

            deconv1_2 = deconv(deconv2_1, 3, 64, 1, 'deconv1_2')
            deconv1_1 = deconv(deconv1_2, 3, self.config.input_depth, 1, 'deconv1_1', tanh=True)
            return deconv1_1
    
    def init_pretrained_weights(self, sess):
        if self.config.fix_weight_layers:
            weights_dict = np.load(Models + 'vgg16.npy', encoding='bytes').item()
            weights_dict = { key.decode('ascii') : value for key, value in weights_dict.items() }
            with tf.variable_scope('generator/f_img', reuse=True):
                num_sublayers = [2, 2, 3, 3]
                for i in range(self.config.fix_weight_layers):
                    layer = i + 1
                    for sublayer in range(1, num_sublayers[i] + 1):
                        name = 'conv%s_%s' % (layer, sublayer)
                        with tf.variable_scope(name):
                            W_value, b_value = weights_dict[name]
                            sess.run(tf.get_variable('W').assign(W_value))
                            sess.run(tf.get_variable('b').assign(b_value))

    def fit_batch_gen(self, feed_dict):
        _, loss, summs = self.sess.run((self.opt_gen, self.loss_gen, self.summs_gen), feed_dict=feed_dict)
        return loss, summs

    def fit_batch_disc(self, feed_dict):
        _, loss, summs = self.sess.run((self.opt_disc, self.loss_disc, self.summs_disc), feed_dict=feed_dict)
        return loss, summs
    
    def predict(self, feed_dict):
        return self.sess.run(self.f_t_n_pred, feed_dict=feed_dict)

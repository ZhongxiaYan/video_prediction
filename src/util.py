from __future__ import print_function, division

import numpy as np

import tensorflow as tf
import scipy
import scipy.io

import os, subprocess

Src = os.path.dirname(os.path.abspath(__file__)) # src directory
Root = os.path.dirname(Src) + '/' # root directory
Src = Src + '/'
Data = os.path.join(Root, 'data') + '/'
Models = os.path.join(Root, 'models') + '/'
Results = os.path.join(Root, 'results') + '/'

def lrelu(x, alpha):
    return tf.maximum(x, alpha*x)

def conv(x, filter_size, num_filters, stride, name, padding='SAME', groups=1, trainable=True):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    regularizer = tf.contrib.layers.l2_regularizer(1e-6)
    with tf.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('W',
                                  shape=[filter_size, filter_size, input_channels // groups, num_filters],
                                  initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer,
                                  trainable=trainable)
        biases = tf.get_variable('b', shape=[num_filters], trainable=trainable, initializer=tf.zeros_initializer())

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)

        return lrelu(conv + biases, 0.2)

def deconv(x, filter_size, num_filters, stride, name, padding='SAME', relu=True, tanh=False):
    activation = None
    if relu:
        activation = tf.nn.relu
    if tanh:
        activation = tf.nn.tanh
    regularizer = tf.contrib.layers.l2_regularizer(1e-6)
    return tf.layers.conv2d_transpose(x, num_filters, filter_size, stride, padding=padding, kernel_regularizer=regularizer, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=activation, name=name)
    
def fc(x, num_out, name, relu=True, trainable=True):
    num_in = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        weights = tf.get_variable('W', shape=[num_in, num_out], initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
        biases = tf.get_variable('b', [num_out], initializer=tf.zeros_initializer(), trainable=trainable)
        x = tf.matmul(x, weights) + biases
        if relu:
            x = lrelu(x, 0.2) 
    return x

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)

def max_pool(x, filter_size, stride, name=None, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

def l2_loss(x, y):
    return tf.reduce_mean(tf.squared_difference(x, y))

def maxout(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    return outputs

def vgg(input, process_input=True):
    if process_input:
        VGG_MEAN = [103.939, 116.779, 123.68]
        
        # Convert RGB to BGR and subtract mean
        red, green, blue = tf.split(input, 3, axis=3)
        input = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], axis=3)
        
    pool_ = lambda x: max_pool(x, 2, 2)
    conv_ = lambda x, output_depth, name: conv(x, 3, output_depth, 1, name=name)
    
    conv1_1 = conv_(input, 64, 'conv1_1')
    conv1_2 = conv_(conv1_1, 64, 'conv1_2')
    pool1 = pool_(conv1_2)

    conv2_1 = conv_(pool1, 128, 'conv2_1')
    conv2_2 = conv_(conv2_1, 128, 'conv2_2')
    pool2 = pool_(conv2_2)

    conv3_1 = conv_(pool2, 256, 'conv3_1')
    conv3_2 = conv_(conv3_1, 256, 'conv3_2')
    conv3_3 = conv_(conv3_2, 256, 'conv3_3')
    pool3 = pool_(conv3_3)

    conv4_1 = conv_(pool3, 512, 'conv4_1')
    conv4_2 = conv_(conv4_1, 512, 'conv4_2')
    conv4_3 = conv_(conv4_2, 512, 'conv4_3')
    pool4 = pool_(conv4_3)

    conv5_1 = conv_(pool4, 512, 'conv5_1')
    conv5_2 = conv_(conv5_1, 512, 'conv5_2')
    conv5_3 = conv_(conv5_2, 512, 'conv5_3')
    pool5 = pool_(conv5_3)
    flattened = tf.contrib.layers.flatten(pool5)

    fc_6 = dropout(fc(flattened, 4096, 'fc6'), 0.5)
    fc_7 = fc(fc_6, 4096, 'fc7', relu=False)
    return fc_7

def vgg_simple(input):
    pool_ = lambda x: max_pool(x, 2, 2)
    conv_ = lambda x, output_depth, name: conv(x, 3, output_depth, 1, name=name)
    
    conv1_1 = conv_(input, 16, 'conv1_1')
    pool1 = pool_(conv1_1)
    conv2_1 = conv_(pool1, 32, 'conv2_1')
    pool2 = pool_(conv2_1)
    conv3_1 = conv_(pool2, 64, 'conv3_1')
    pool3 = pool_(conv3_1)
    conv4_1 = conv_(pool3, 64, 'conv4_1')
    pool4 = pool_(conv4_1)
    conv5_1 = conv_(pool4, 64, 'conv5_1')
    pool5 = pool_(conv5_1)
    
    flattened = tf.contrib.layers.flatten(pool5)
    fc_6 = dropout(fc(flattened, 4096, 'fc6'), 0.5)
    fc_7 = fc(fc_6, 4096, 'fc7', relu=False)
    return fc_7

def vgg_deconv(input):
    deconv6_2 = deconv(input, 7, 128, 1, 'deconv6_2', padding='VALID')
    deconv6_1 = deconv(deconv6_2, 3, 128, 2, 'deconv6_1')

    deconv5_2 = deconv(deconv6_1, 3, 128, 1, 'deconv5_2')
    deconv5_1 = deconv(deconv5_2, 3, 128, 2, 'deconv5_1')

    deconv4_3 = deconv(deconv5_1, 3, 128, 1, 'deconv4_3')
    deconv4_2 = deconv(deconv4_3, 3, 128, 1, 'deconv4_2')
    deconv4_1 = deconv(deconv4_2, 3, 64, 2, 'deconv4_1')

    deconv3_3 = deconv(deconv4_1, 3, 64, 1, 'deconv3_3')
    deconv3_2 = deconv(deconv3_3, 3, 64, 1, 'deconv3_2')
    deconv3_1 = deconv(deconv3_2, 3, 32, 2, 'deconv3_1')

    deconv2_2 = deconv(deconv3_1, 3, 32, 1, 'deconv2_2')
    deconv2_1 = deconv(deconv2_2, 3, 16, 2, 'deconv2_1')

    deconv1_2 = deconv(deconv2_1, 3, 16, 1, 'deconv1_2')
    deconv1_1 = deconv(deconv1_2, 3, 3, 1, 'deconv1_1')
    return deconv1_1

def alexnet(input):
    # input: 227 x 227 x 3 rgb image. 0 to 255
    input_norm = input - tf.reduce_mean(input)
    bgr = tf.reverse(input_norm, axis=3)
    
    def conv(x, filter_size, num_filters, stride, name, groups=1, padding='SAME', relu=True):
        input_channels = int(x.get_shape()[-1])

        convolve = lambda x, W: tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        with tf.variable_scope(name):
            weights = tf.get_variable('W',
                                      shape=[filter_size, filter_size, input_channels // groups, num_filters],
                                      trainable=False)
            biases = tf.get_variable('b', shape=[num_filters], trainable=False)

            if groups == 1:
                output = convolve(x, weights)
            else:
                input_groups = tf.split(x, groups, axis=3)
                weight_groups = tf.split(weights, groups, axis=3)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
                output = tf.concat(output_groups, axis=3)
                
            output = tf.reshape(tf.nn.bias_add(output, biases), output.get_shape().as_list())
            if relu:
                return tf.nn.relu(output)
            return output

    conv1 = conv(bgr, 11, 96, 4, name='conv1', padding='VALID')
    pool1 = max_pool(conv1, 3, 2, padding='VALID', name='pool1')
    norm1 = lrn(pool1, 2, 2e-5, 0.75, name='norm1')

    conv2 = conv(norm1, 5, 256, 1, groups=2, name='conv2')
    pool2 = max_pool(conv2, 3, 2, padding='VALID', name='pool2')
    norm2 = lrn(pool2, 2, 2e-5, 0.75, name='norm2')

    conv3 = conv(norm2, 3, 384, 1, name='conv3')
    conv4 = conv(conv3, 3, 384, 1, groups=2, name='conv4')
    conv5 = conv(conv4, 3, 256, 1, groups=2, name='conv5', relu=False)
    return conv5

def get_minibatch(batch_size, videos, poses):
    batch_data = []
    video_ids = list(videos.keys())
    n_videos = len(video_ids)
    for _ in range(batch_size):
        v_i = video_ids[np.random.randint(n_videos)]
        video = videos[v_i]
        pose = poses[v_i]
        n_frames = len(video)
        f_i1, f_i2 = np.random.randint(0, n_frames, size=2)
        batch_data.append((video[f_i1], pose[f_i1], video[f_i2], pose[f_i2]))
    return zip(*batch_data)

def generate_heatmap(x, y, w_orig, h_orig, zeros, sigma=5, w_new=224, h_new=224):
    sigma_y = sigma * h_new / h_orig
    sigma_x = sigma * w_new / w_orig
    y_radius = int(sigma_y * 3) + 1
    x_radius = int(sigma_x * 3) + 1
    x_int = int(np.round(x))
    y_int = int(np.round(y))
    x_start, x_end = max(0, x_int - x_radius), min(w_new, x_int + x_radius + 1)
    y_start, y_end = max(0, y_int - y_radius), min(h_new, y_int + y_radius + 1)
    xx, yy = np.meshgrid(np.arange(x_start, x_end), np.arange(y_start, y_end))
    zz = np.exp(-0.5 * ((xx - x) ** 2 / (sigma_x ** 2) + (yy - y) ** 2 / (sigma_y ** 2))) * 255

    zeros[y_start : y_end, x_start : x_end] += zz.astype(np.uint8)

def load_annotations():
    annotations = {}
    for name, path in list_dir(Data + 'labels/', 'mat', return_name=True):
        data = scipy.io.loadmat(path)
        h, w, n = data['dimensions'][0]
        base_bbox = np.round(data['bbox']).astype(int)
        
        # generates bbox per frame
        padding = 30
        bbox = base_bbox + np.array([[-padding, -padding, padding, padding]])
        bbox = np.maximum(bbox, 0)
        bbox[:, 2:] = np.minimum(bbox[:, 2:], np.array([[w, h]]))
        
        # generates same bbox for all frames
        same_bbox = np.concatenate((np.min(base_bbox[:, :2], axis=0), np.max(base_bbox[:, 2:], axis=0)))
        same_bbox = np.maximum(same_bbox, 0)
        same_bbox[2:] = np.minimum(same_bbox[2:], np.array([w, h]))
        
        annotations[name] = {
            'action' : data['action'][0].encode('ascii'),
            'bbox' : bbox,
            'same_bbox' : same_bbox,
            'coords' : np.concatenate((np.expand_dims(data['x'], 2), np.expand_dims(data['y'], 2)), axis=2),
            'dimensions' : (h, w),
            'nframes' : n,
            'pose' : data['pose'][0].encode('ascii'),
            'train' : data['train'][0, 0] == 1,
            'visibility' : data['visibility'],
        }
    return annotations
    
def list_dir(dir, ext, return_name=False):
    ext = '.' + ext.lower()
    if return_name:
        return sorted([(file[:-len(ext)], dir + file) for file in os.listdir(dir) if file[-len(ext):].lower() == ext])
    else:
        return sorted([dir + file for file in os.listdir(dir) if file[-len(ext):].lower() == ext])

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_name(path):
    return '.'.join(path.split('/')[-1].split('.')[:-1])
        
def remove(path):
    if not os.path.exists(path):
        return
    elif os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)

def shell(cmd, wait=True, ignore_error=True):
    if type(cmd) != str:
        cmd = ' '.join(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not wait:
        return process
    out, err = process.communicate()
    if err and not ignore_error:
        print(err.decode('UTF-8'))
        raise RuntimeError('Error in command line call')
    return out.decode('UTF-8'), err.decode('UTF-8') if err else None

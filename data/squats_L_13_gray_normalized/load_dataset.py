import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat

import os

from util import *

def load_dataset(type):
    assert type in ['train', 'val', 'test']
    type_dir = os.path.join(Data, 'squats_L_1', type) + '/'
    labels_dir = os.path.join(Data, 'labels') + '/'
    video_names = os.listdir(type_dir)
    videos = {}
    poses = {}
    for v_name in video_names:
        v = np.load(type_dir + v_name + '/video.npy')
        new_v = [np.array(Image.fromarray(f).convert('L')).reshape((224, 224, 1)) for f in v]
        videos[v_name] = np.array(new_v)
        annotations = loadmat(labels_dir + v_name + '.mat')
        visible = annotations['visibility']
        h, w, n = annotations['dimensions'][0]
        pose = np.zeros((n, 224, 224, 13), dtype=np.uint8)

        xs = annotations['x'] * 224 / w
        ys = annotations['y'] * 224 / h

        for f_i in range(n):
            for p_i in range(13):
                if not visible[f_i][p_i]:
                    continue
                x, y = xs[f_i][p_i], ys[f_i][p_i]
                generate_heatmap(x, y, w, h, pose[f_i, :, :, p_i]) # from util
        poses[v_name] = pose
    return videos, poses

def save_predictions(bundle, output_dir):
    make_dir(output_dir)
    for i, bundle_i in enumerate(bundle):
        output_i = os.path.join(output_dir, str(i)) + '.png'
        N = len(bundle_i)
        plt.figure(figsize=(20, 20))
        for j, img in enumerate(bundle_i):
            plt.subplot(1, N, j + 1)

            if img.shape[2] == 1:
                img = img.reshape((224, 224))
            if img.shape[2] == 13:
                img = np.sum(img, axis = 2)
            print(img.shape)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(output_i)
            

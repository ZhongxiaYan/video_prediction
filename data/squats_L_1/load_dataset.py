import numpy as np
import matplotlib.pyplot as plt

import os

from util import *

dataset_directory = os.path.dirname(__file__)

def load_dataset(type):
    assert type in ['train', 'val', 'test']
    type_dir = os.path.join(dataset_directory, type) + '/'
    video_names = os.listdir(type_dir)
    videos = {}
    poses = {}
    for v_name in video_names:
        v = np.load(type_dir + v_name + '/video.npy')
        p = np.load(type_dir + v_name + '/poses_L_1.npy')
        videos[v_name] = v
        poses[v_name] = p
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
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(output_i)
            
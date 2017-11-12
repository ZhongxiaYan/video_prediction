import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os

from util import *

dataset_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'squats_L_1')

def load_dataset(type):
    assert type in ['train', 'val', 'test']
    type_dir = os.path.join(dataset_directory, type) + '/'
    video_names = os.listdir(type_dir)
    videos = {}
    poses = {}
    for v_name in video_names:
        v = np.load(type_dir + v_name + '/video.npy')
        new_v = []
        for frame in v:
            new_v.append(np.array(Image.fromarray(frame).convert('L')).reshape((224, 224, 1)))
        p = np.load(type_dir + v_name + '/poses_L_1.npy')
        videos[v_name] = np.array(new_v)
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
            
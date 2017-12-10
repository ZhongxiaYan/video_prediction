import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from PIL import Image
import imageio

import os

from util import *

class PennAction:
    def __init__(self, config):
        self.config = config
        self.annotations = load_annotations()
        self.videos = {}

        all_actions = set([vinfo['action'] for vinfo in self.annotations.values()])
        
        if config.actions == 'all':
            actions = all_actions
        else:
            for action in config.actions:
                assert action in all_actions, 'Action "%s" is not valid' % action
            actions = set(config.actions)
        self.actions = actions
        self.train_videos = [vname for vname, vinfo in self.annotations.iteritems() if vinfo['train'] and vinfo['action'] in actions]
        self.train_action_to_samples = self.get_action_to_samples(train=True)
        self.test_action_to_samples = self.get_action_to_samples(train=False)
    
    def get_video(self, vname):
        config = self.config
        if vname not in self.videos:
            if config.cropped:
                if config.cropped == 'consistent':
                    vdir = 'videos_same_bbox_224'
                else:
                    vdir = 'videos_bbox_224'
            else:
                vdir = 'videos_224'
            video = np.load(Data + vdir + '/' + vname + '.npy')
            if config.gray:
                video = np.array([np.array(Image.fromarray(f).convert('L')).reshape((224, 224, 1)) for f in video])
            self.videos[vname] = video
        return self.videos[vname]
    
    def get_pose(self, vinfo, frame_num):
        config = self.config
        if config.cropped:
            if config.cropped == 'consistent':
                minx, miny, maxx, maxy = vinfo['same_bbox']
            else:
                minx, miny, maxx, maxy = vinfo['bbox'][frame_num]
            w = maxx - minx
            h = maxy - miny
        else:
            minx = miny = 0
            h, w = vinfo['dimensions']
            
        pose = np.zeros((224, 224, 13), dtype=np.float32)
        for i, ((x, y), visible) in enumerate(zip(vinfo['coords'][frame_num], vinfo['visibility'][frame_num])):
            if visible:
                x_new = (x - minx) * 224.0 / w
                y_new = (y - miny) * 224.0 / h
                generate_heatmap(x_new, y_new, w, h, pose[:, :, i], sigma=5, w_new=224, h_new=224)
        if config.L == 1:
            return np.sum(pose, axis=2, keep_dims=True)
        else:
            return pose
    
    def _get_batch(self, video_batch):
        batch_data = []
        for vname in video_batch:
            vinfo = self.annotations[vname]
            video = self.get_video(vname)
            n = min(vinfo['nframes'], len(vinfo['bbox']))
            if 'min_frame_dist' in self.config:
                min_frame_dist = min(self.config.min_frame_dist, n // 2)
            else:
                min_frame_dist = 0
            f_i1, f_i2 = np.random.randint(0, n, size=2)
            while abs(f_i1 - f_i2) < min_frame_dist:
                f_i1, f_i2 = np.random.randint(0, n, size=2)
            pose1 = self.get_pose(vinfo, f_i1)
            pose2 = self.get_pose(vinfo, f_i2)
            if np.random.random() > 0.5:
                batch_data.append((video[f_i1], pose1, video[f_i2], pose2))
            else:
                batch_data.append(tuple(np.flip(x, axis=1) for x in (video[f_i1], pose1, video[f_i2], pose2)))
        return zip(*batch_data)
        
    def get_train_batch(self, batch_size):
        video_batch = np.random.choice(self.train_videos, size=batch_size, replace=True)
        return self._get_batch(video_batch)
    
    def get_action_to_samples(self, train):
        action_to_samples = {}
        for vname, vinfo in self.annotations.iteritems():
            if vinfo['train'] == train and vinfo['action'] in self.actions:
                vnames = action_to_samples.setdefault(vinfo['action'], [])
                if len(vnames) < self.config.N_test_per_class:
                    vnames.append(vname)
        return action_to_samples
    
    def get_val_batch(self, test=True):
        if test:
            action_to_samples = self.test_action_to_samples
        else:
            action_to_samples = self.train_action_to_samples
        old_state = np.random.get_state()
        np.random.seed(0)
        for action, vnames in action_to_samples.iteritems():
            yield action, vnames, self._get_batch(vnames)
        np.random.set_state(old_state)
    
    def get_test_batch(self, test=True):
        if test:
            action_to_samples = self.test_action_to_samples
        else:
            action_to_samples = self.train_action_to_samples
        for action, vnames in action_to_samples.iteritems():
            videos, poses = [], []
            for vname in vnames:
                vinfo = self.annotations[vname]
                videos.append(self.get_video(vname))
                poses.append(np.array([self.get_pose(vinfo, i) for i in range(vinfo['nframes'])]))
            yield action, vnames, videos, poses
    
    def save_video_predictions(self, output_dir, action, vname, video):
        imageio.mimsave(os.path.join(output_dir, action, vname) + '.gif', video)
    
    def save_image_predictions(self, output_dir, action, vname, f1, p1, f2, p2, pred):
        plt.figure(figsize=(15, 15))
        def subplot_frame(i, f):
            plt.subplot(1, 5, i)
            if self.config.gray:
                plt.imshow(f.reshape((224, 224)), cmap='gray')
            else:
                plt.imshow(f)
            plt.axis('off')
        def subplot_pose(i, p):
            plt.subplot(1, 5, i)
            plt.imshow(np.sum(p, axis=2), cmap='gray')
            plt.axis('off')
        subplot_frame(1, f1)
        subplot_pose(2, p1)
        subplot_frame(3, f2)
        subplot_pose(4, p2)
        subplot_frame(5, pred)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(os.path.join(output_dir, action, vname) + '.png', bbox_inches='tight')
        plt.close()
        
def load_dataset(config):
    return PennAction(config)
        
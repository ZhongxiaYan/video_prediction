import os

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

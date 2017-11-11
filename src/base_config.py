from easydict import EasyDict as edict
from util import *

def base_config():
	config = edict()
	
	# model parameters
	config.pretrained_models = {}
	config.lr_gen = 1e-4
	config.lr_disc = 1e-4
	config.keep_prob = None
	config.adv_coef = 1

	# training parameters
	config.batch_size

	# input parameters
	config.image_width
	config.image_height

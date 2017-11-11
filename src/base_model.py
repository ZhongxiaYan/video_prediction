import numpy as np
import tensorflow as tf

class NNBase:
	def __init__(self, config, sess):
		self.config = config
		self.sess = sess

	def train(self, saver, summary_writer, train_data, val_data, train_dir):
		pass

	def test(self, test_data):
		pass
		
	def get_train_variables(self):
		return tf.global_variables()

	def get_test_variables(self):
		return tf.global_variables()
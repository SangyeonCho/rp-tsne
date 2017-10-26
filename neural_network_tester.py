# Copyright (C) James Dolezal - All Rights Reserved
# Written by James Dolezal <jmd172@pitt.edu>, July 2017
# ==========================================================================
#
# Neural network trainer
#  Evaluates performance of a saved Tensorflow neural network model on a test data set.
#
# Use:
#  - Requires Tensorflow r1.0 ( https://www.tensorflow.org ) and numpy
#  - Input data (for either training or testing) should contain outputs in first column (e.g. Cluster number)
#     and data in remaining columns (e.g. ribosomal protein transcript relative expression)

import argparse
import sys
import os
import numpy as np
from numpy import genfromtxt

from tensorflow.contrib.tensorboard.plugins import projector

import tensorflow as tf

FLAGS = None

def convertOneHot(data):
	# Converts array of numbers into array of onehot numbers
	y = np.array([int(i[0]) for i in data])
	y_onehot = [0] * len(y)
	for i,j in enumerate(y):
		y_onehot[i] = [0] * (y.max() + 1)
		y_onehot[i][j] = 1
	return (y, y_onehot)

def eval(file):
	test_data = genfromtxt(FLAGS.data_dir + '/%s' % file, delimiter=',') # test data

	x_test = np.array([ i[1::] for i in test_data])
	y_test, y_test_onehot = convertOneHot(test_data)

	with tf.Session() as sess:

		model = FLAGS.model_dir + '/' + FLAGS.model + ".meta"
		print("Importing model %s from: %s ..." % (FLAGS.model, model))
		loader = tf.train.import_meta_graph(model)
		print("Restoring imported model...")
		loader.restore(sess, FLAGS.model_dir + '/' + FLAGS.model)

		accuracy = tf.get_collection('accuracy')

		acc = sess.run(accuracy, feed_dict = {"x-input:0": x_test, "y-input:0":y_test_onehot, "dropout/keep_prob:0":1})
		print("Model accuracy: %s" % acc[0])

def main(_):
	print("Loading data from test file %s" % FLAGS.file)
	eval(FLAGS.file)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--data_dir', type=str, help='Directory for storing input data')
	parser.add_argument('--model_dir', type=str, help='Model location')
	parser.add_argument('--file', required=True, help='Training data filename')
	parser.add_argument('--model', required=True, help='Model to load')
	
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

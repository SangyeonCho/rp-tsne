# Copyright (C) James Dolezal - All Rights Reserved
# Written by James Dolezal <jmd172@pitt.edu>, July 2017
# ==========================================================================
#
# Neural network trainer
#  Trains a Tensorflow neural network model on a training data set.
#
# Use:
#  - Requires Tensorflow r1.0 ( https://www.tensorflow.org ) and numpy
#  - Input data (for either training or testing) should contain outputs in first column (e.g. Cluster number)
#     and data in remaining columns (e.g. ribosomal protein transcript relative expression)

import argparse, sys, os
import numpy as np 
import tensorflow as tf

from numpy import genfromtxt

FLAGS = None

TRANSPOSE = False
NETWORK_STRUCTURE = [120]

def convertOneHot(data):
	# Converts array of numbers from first column in data into array of onehot numbers (for nn output)

	y = np.array([int(i[0]) for i in data])
	y_onehot = [0] * len(y)
	for i, j in enumerate(y):
		y_onehot[i] = [0] * (y.max() + 1)
		y_onehot[i][j] = 1
	return y_onehot

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act):
	with tf.variable_scope(layer_name):
		weights = tf.get_variable("weights", [input_dim, output_dim], initializer = tf.random_normal_initializer())
		biases = tf.get_variable("biases", [output_dim], initializer = tf.constant_initializer(0.01))
		preactivate = tf.matmul(input_tensor, weights) + biases
		activations = act(preactivate, name = 'activation')
		return activations, weights

def generate_layers(x, A, B, keep_prob, hidden_size):
	# Generates hidden layers of size according to array hidden_size

	layers, weights = [], []

	if len(hidden_size) == 0:
		layer, w = nn_layer(x, A, B, 'layer1', tf.identity)
		weights.append(w)
		return layer, weights

	for i in range(len(hidden_size) + 1):
		prev_tens = x if i == 0 else layers[i-1]
		prev_size = A if i == 0 else hidden_size[i-1]
		curr_size = B if i == len(hidden_size) else hidden_size[i]
		act = tf.identity if i == len(hidden_size) else tf.nn.relu

		layer, w = nn_layer(prev_tens, prev_size, curr_size, 'layer%s' % i, act)
		weights.append(w)

		with tf.name_scope('dropout'):
			dropped = tf.nn.dropout(layer, keep_prob)
		layers.append(dropped)

	return layers[-1], weights

def train():

	tf.reset_default_graph()

	data = genfromtxt(FLAGS.data_dir + '/' + FLAGS.train_file, delimiter = ',') # training data (60%)
	test_data = genfromtxt(FLAGS.data_dir + '/' + FLAGS.test_file, delimiter = ',') # validation data (10%)

	if TRANSPOSE:
		data = data.transpose() #map(data, zip(*data))
		test_data = test_data.transpose() #map(test_data, zip(*test_data))

	x_train = np.array([ i[1::] for i in data])
	y_train_onehot = convertOneHot(data)

	x_test = np.array([ i[1::] for i in test_data])
	y_test_onehot = convertOneHot(test_data)

	A = data.shape[1] - 1 # number of features
	B = len(y_train_onehot[0]) # number of output categories

	data_size = x_train.shape[0]
	merge_end = (data_size % FLAGS.batch_size) < (FLAGS.batch_size / 2)
	num_batches = int(data_size/FLAGS.batch_size) + (0 if merge_end else 1)

	batches = []
	for i in range(num_batches):
		if i != num_batches-1:
			xs = x_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
			ys = y_train_onehot[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
		else:
			xs = x_train[i*FLAGS.batch_size:]
			ys = y_train_onehot[i*FLAGS.batch_size:]
		batches.append([xs, ys])

	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32, name="keep_prob")

	with tf.Session() as sess:

		x = tf.placeholder(tf.float32, [None, A], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, B], name='y-input')

		y, weights = generate_layers(x, A, B, keep_prob, NETWORK_STRUCTURE)
		
		with tf.name_scope('cross_entropy'):
			diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
			with tf.name_scope('total'):
				lossL2 = 0#tf.add_n([tf.nn.l2_loss(v) for v in weights]) * 0.0001
				cross_entropy = tf.reduce_mean(diff + lossL2)

		with tf.name_scope('train'):
			train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)

		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		def feed_dict(train, step = 0):
			batch = batches[step % num_batches]
			xs = batch[0] if train else x_test
			ys = batch[1] if train else y_test_onehot
			k = FLAGS.dropout if train else 1.0
			return {x: xs, y_: ys, keep_prob: k}

		tf.global_variables_initializer().run()
		saver = tf.train.Saver()
		tf.add_to_collection('accuracy', accuracy)
		saved = False

		os.system('cls' if os.name == 'nt' else 'clear')

		print("\nData")
		print("----")
		print("Training data:       %s" % FLAGS.train_file)
		print("Validation data:     %s" % FLAGS.test_file)
		print("Training data size:  %s" % data_size)
		print("Model:               %s\n" % FLAGS.model)

		print("\nNetwork parameters:")
		print("-"*19)
		print("Network structure:   %s" % NETWORK_STRUCTURE)
		print("Learning rate:       %s" % FLAGS.learning_rate)
		print("Regularization:      ReLU")
		print("Batch size:          %s x %s batches" % (FLAGS.batch_size, num_batches))
		print("Dropout:             %s" % FLAGS.dropout)
		print("Epochs:              %s" % FLAGS.max_steps)
		print("Accuracy goal:       %s \n\n" % FLAGS.acc_goal)

		max_accuracy = 0
		step = 0

		print("Training:\n---------")

		try:
			while step < FLAGS.max_steps * num_batches:
				if step % (50 * min(num_batches, 20)) == 0:
					# run training and accuracy
					sess.run(train_step, feed_dict = feed_dict(True, step))
					#print('Step %s' % step)
					acc = sess.run(accuracy, feed_dict = feed_dict(False))
					max_accuracy = max(acc, max_accuracy)
					epoch = str(int(step / num_batches))
					sys.stdout.write('\rAccuracy at epoch %s: %.1f (max: %.1f)\t' % (' '*(6-len(epoch)) + epoch, acc*100, max_accuracy*100))
					progressBar(step, FLAGS.max_steps*num_batches)

					if acc >= FLAGS.acc_goal:
						print('\nAccuracy reached %s, saving model...' % acc)
						saver.save(sess, os.path.join(FLAGS.save_dir + '/' + FLAGS.model))
						saved = True
						print('Exiting...')
						break
					step += 1
				else:
					# just run training
					sess.run(train_step, feed_dict = feed_dict(True, step))
					step += 1
		except (KeyboardInterrupt, SystemExit):
			acc = sess.run(accuracy, feed_dict = feed_dict(False))
			print('\nSaving model "%s" with %.1f' % (FLAGS.model, acc*100) + '%' + " accuracy")
			saver.save(sess, os.path.join(FLAGS.save_dir + '/' + FLAGS.model))
			sys.exit()

		if not saved:
			acc = sess.run(accuracy, feed_dict = feed_dict(False))
			saver.save(sess, os.path.join(FLAGS.save_dir + '/' + FLAGS.model))
			print('\nModel "%s" saved with %s accuracy, exiting...' % (FLAGS.model, acc))

def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	train()

def progressBar(value, endvalue, bar_length=20):

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("[{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--max_steps', type=int, default=20000, help='Number of epochs to run trainer')
	parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
	parser.add_argument('--learning_rate', type=float, default = 0.002, help='Initial learning rate')
	parser.add_argument('--dropout', type=float, default = .9, help='Keep probability for training dropout')
	parser.add_argument('--acc_goal', type=float, default = 1, help='Accuracy goal to trigger training completion and model saving')
	parser.add_argument('--data_dir', type=str, help="Directory for storing input data")
	parser.add_argument('--log_dir', type=str, help = 'Summaries log directory')
	parser.add_argument('--save_dir', type=str, help = 'Model save directory')
	parser.add_argument('--train_file', required=True, help = 'Training data filename')
	parser.add_argument('--test_file', required=True, help = 'Validation data filename')
	parser.add_argument('--model', required=True, help = 'Model name')
	
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

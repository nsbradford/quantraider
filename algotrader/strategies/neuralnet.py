#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


BATCH_SIZE = 1
SEQ_LENGTH = 1774 #1775
FEATURES_SIZE = 4 #(day, high-open, close-open, low-open)
OUTPUT_SIZE = 1
num_epochs=1000

def load_dataset():
	# load data from csv file
	import csv
	f = open('usdjpy5year-dhloc.csv');
	raw_data = list(csv.reader(f)); # read in data
	
	# pre-process data (day, high, low, open, close) -> (day, high-open, close-open, low-open)
	data = [];
	for row in raw_data:
		t = [];
		t.append(int(row[0])); #day
		c = float(row[3]); # open
		t.append(float(row[1]) - c); #high - open
		t.append(float(row[4]) - c); #close - open
		t.append(float(row[2]) - c); #low - open
		data.append(t);
	
	# expected output
	target = []
	for i in range(len(data)):
		if (i == 0):
			continue;
		row = data[i];
		#print(row);
		t = [0];
		if row[2] > 0:
			t[0] = 1;		
		target.append(t); # list with one element, one for high, or zero for low
		
	del data[-1]; # delete to shift target and data over to account for predicting next day, not today
	#print(target);
	#return 0;
	
	# normalize data by scaling between -1 and 1
	# makes search space smaller for the pattern detection
	m = np.matrix(data);
	for i in range(np.size(m, 1)):
		d = m[:, i];
		m[:, i] = -1 + 2 * ((d - min(d))/(max(d) - min(d)))
	
	#SEQ_LENGTH = len(data);
	# input (x, y) pairs as input, expected output
	X_train = np.reshape(m.getA1(), (BATCH_SIZE, SEQ_LENGTH, FEATURES_SIZE));
	X_train = X_train.astype(theano.config.floatX);
	
	y_train = np.reshape(np.matrix(target).getA1(), (BATCH_SIZE, SEQ_LENGTH, OUTPUT_SIZE));
	y_train = y_train.astype(theano.config.floatX);
	return X_train, y_train, X_train, y_train;
	
	# not useful
	X_val = X_train[-1].reshape((1, SEQ_LENGTH, FEATURES_SIZE));
	y_val = y_train[-1].reshape((1, SEQ_LENGTH, OUTPUT_SIZE));
	X_train = X_train[:-1];
	y_train = y_train[:-1];
	return X_train, y_train, X_val, y_val

def main(model='mlp'):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val = load_dataset()

	# Prepare Theano variables for inputs and targets
	#input_var = T.tensor3('inputs')
	target_var = T.tensor3('targets')

	print("Building model and compiling functions...")
	l_in = lasagne.layers.InputLayer((None, SEQ_LENGTH, FEATURES_SIZE))

	#TODO: use adam, gru, and steeper activation
	num_units = 50 # number of units in LSTM layer
	network = lasagne.layers.LSTMLayer(l_in, num_units=num_units, learn_init=True, grad_clipping=5)
	network = lasagne.layers.ReshapeLayer(network, (-1, num_units))
	
	network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),num_units=150,nonlinearity=lasagne.nonlinearities.rectify)
	
	num_classes = 1
	batchsize, seqlen, _ = l_in.input_var.shape
	network = lasagne.layers.DenseLayer(network, num_units=num_classes, nonlinearity=lasagne.nonlinearities.sigmoid)
	network = lasagne.layers.ReshapeLayer(network, (batchsize, seqlen, num_classes))
	
	print("Total number of network parameters: ", lasagne.layers.count_params(network));
	
	# determine loss function
	# not deterministic because Dropout drops a random number of neuron outputs in each layer
	# helps with overfitting
	# only use during training, test with deterministic=True
	prediction = lasagne.layers.get_output(network, deterministic=False) 
	loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
	
	#regularization, l2 and l1 and dropout are other options
	regu = 0.00005
	regu_loss = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
	loss = loss + regu * regu_loss
	loss = loss.mean()
	params = lasagne.layers.get_all_params(network, trainable=True)
	
	#updates = lasagne.updates.adagrad(loss, params, 0.01)
	#updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
	#updates is a function
	updates = lasagne.updates.adam(loss, params, learning_rate=0.001) #training method (not gradient descent)

	# from example. train_fn is a pointer to a function. decide that you return loss
	train_fn = theano.function([l_in.input_var, target_var], loss, updates=updates) # remove updates=updates to predict without training
	
	# same thing as above, but without regularization/dropout
	val_pred = lasagne.layers.get_output(network, deterministic=True)
	val_loss = lasagne.objectives.binary_crossentropy(val_pred, target_var).mean();
	val_acc = T.mean(T.eq(T.round(val_pred), target_var))
	val_fn = theano.function([l_in.input_var, target_var], [val_loss, val_acc]) #decide what you want value function to return
	max_acc = 0 # max accuracy
	print("Training ...")
	try:
		for epoch in range(num_epochs):
			for _ in range(10):
				cost_train = train_fn(X_train, y_train) #result of binary_crossentropy loss function
				print("Training cost = {}".format(cost_train))
			cost_val, acc_val = val_fn(X_val, y_val) 
			if acc_val > max_acc:
				max_acc = acc_val
				
			if acc_val == 0.0:
				break;
			print("Epoch {} validation cost = {}, accuracy = {}, max_acc = {}".format(epoch, cost_val, acc_val, max_acc))
	except KeyboardInterrupt:
		pass

main();
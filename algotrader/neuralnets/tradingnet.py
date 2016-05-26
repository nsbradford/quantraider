import numpy as np
import theano
import theano.tensor as T
import lasagne
import time

class TradingNet(object):
	'''
	Accepts arguments
		x: list of input data in the form
			x = [
				[[x1 x2 ... xn],[x1 x2 ... xn],[x1 x2 ... xn]], # stock 1
				[[x1 x2 ... xn],[x1 x2 ... xn],[x1 x2 ... xn]]  # stock 2
			]
		y: list of expected data in the form. all values of y must be 0 or 1. NOTE: currently only supports a single output y.
			y = [
				[[y1 y2 ... yn],[y1 y2 ... yn],[y1 y2 ... yn]], # stock 1 expected output
				[[y1 y2 ... yn],[y1 y2 ... yn],[y1 y2 ... yn]]  # stock 2 expected output
			]
		All stock data must have the same amount of timesteps/samples
	'''
	def __init__(self, x, y, num_epochs=1000):
		#automatically define num_inputs, num_outputs, and max_length
		if len(x) < 0 or len(x) != len(y):
			raise ValueError("Invalid training data")
			
		num_inputs = len(x[0][0])
		num_outputs = len(y[0][0])
		max_length = len(x[0])
		
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.max_length = max_length
		
		#initialize neural network
		l_in = lasagne.layers.InputLayer((None, self.max_length, num_inputs))
		l_mask = lasagne.layers.InputLayer((None, self.max_length))
		self.input_layer = l_in
		self.mask_layer = l_mask
		
		#TODO: use adam, gru, and steeper activation
		num_units = 64 #number of units in LSTM layer
		network = lasagne.layers.LSTMLayer(l_in, num_units=num_units, mask_input=l_mask, learn_init=True, nonlinearity=lasagne.nonlinearities.tanh)
		
		# need to reshape to use recurrent layer output in standard feedforward
		network = lasagne.layers.ReshapeLayer(network, (-1, num_units))
		network = lasagne.layers.DenseLayer(network,num_units=10, nonlinearity=lasagne.nonlinearities.sigmoid)

		# only outputs a single number between 0 and 1
		
		batchsize, seqlen, _ = l_in.input_var.shape
		network = lasagne.layers.DenseLayer(network, num_units=self.num_outputs, nonlinearity=lasagne.nonlinearities.sigmoid)
		network = lasagne.layers.ReshapeLayer(network, (batchsize, seqlen, self.num_outputs))
		self.network = network
		
		# train on 90% of data
		#train_length = round(self.max_length * 0.9) 
		train_mask = np.ones((len(x), self.max_length))
		#val_mask = np.ones((len(x), self.max_length))
		#for n in range(len(x)):
		#	train_mask[n, train_length:] = 0
		
		
		train_x = np.array(x).reshape((len(x), self.max_length, self.num_inputs))
		train_y = np.array(y).reshape((len(x), self.max_length, self.num_outputs))
		
		#required for theano. not sure why.
		train_x = train_x.astype(theano.config.floatX)
		train_y = train_y.astype(theano.config.floatX)
		train_mask = train_mask.astype(theano.config.floatX)
		
		target_var = T.tensor3('targets')
		prediction = lasagne.layers.get_output(self.network, deterministic=False)

		loss = lasagne.objectives.binary_crossentropy(prediction, target_var)

		#temporary disable regularization
		'''
		regu = 0.00005 
		regu_loss = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
		loss = loss + regu * regu_loss
		'''
		
		loss = loss.mean()
		params = lasagne.layers.get_all_params(self.network, trainable=True)

		updates = lasagne.updates.adam(loss, params, learning_rate=0.001)

		train_fn = theano.function([self.input_layer.input_var, target_var, self.mask_layer.input_var], loss, updates=updates)
		start_time = time.time()
		for epoch in range(num_epochs):
			training_error = train_fn(train_x, train_y, train_mask)
			print ("Epoch {} training error = {}".format(epoch, training_error))
			print "Epochs complete:", str(epoch) + "/" + str(num_epochs), "\tTime elapsed:", str(time.time()-start_time), "seconds."


	'''
		Accepts arguments
		x: a single sequence of input of length similar to training data
		
		Outputs a list of digits between 0 and 1 for all timesteps in the sequence
	'''
	def predict(self, x):
		assert len(x) == self.max_length, "ERROR: " + str(len(x)) + " instead of " + str(self.max_length)
		assert len(x[0]) == self.num_inputs, "ERROR: " + str(len(x[0])) + " instead of " + str(self.num_inputs)
		#print len(x), self.max_length, len(x[0]), self.num_inputs
		pred_x = np.array(x).reshape((1, self.max_length, self.num_inputs))
		pred_x = pred_x.astype(theano.config.floatX)
		pred_mask = np.ones((1, self.max_length)).astype(theano.config.floatX)
		prediction = lasagne.layers.get_output(self.network, deterministic=True)
		pred_fn = theano.function([self.input_layer.input_var, self.mask_layer.input_var], prediction)
		
		ret = pred_fn(pred_x, pred_mask)
		ret = ret.reshape((self.max_length, self.num_outputs))
		return ret

def testNN():
	x = [
		[[0.1],[0.2],[0.3],[0.4],[0.5]],
		[[-0.1],[-0.2],[-0.3],[-0.4],[-0.5]]
	]
	y = [
		[[1],[0],[1],[0],[1]],
		[[1],[0],[1],[0],[1]]
	]
	#In this example...
	#Number of sequences: 2
	#Number of timesteps per sequence: 5
	#Number of inputs per timestep: 1
	testnet = TradingNet(x, y, num_epochs=2)
	testprediction = testnet.predict([[0.1],[0.2],[0.3],[0.4],[0.5]]) #input is one full sequence
	print(testprediction) #expected: [[1],[0],[1],[0],[1]]
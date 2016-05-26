import numpy as np

import sklearn
from sklearn.naive_bayes import GaussianNB
import time

class NaiveBayes(object):
	'''
	Accepts arguments

		[[[open = 0, high, low, close],
		  [                           ]]]


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

		# print str(x)

		self.num_inputs = len(x[0][0])
		self.num_outputs = len(y[0][0])
		self.max_length = len(x[0])

		numInputs = len(x[0])
		# print(numInputs)

		pastFiveDays = [] # each feature i will be [h, l, c, o] for day t-i for i = 1...5
		followingDay = [] # each day will be day t

		for i in range(5, numInputs-1):
			# print str(i)
			pastFiveDays.append(np.reshape(x[0][i-5:i],20))
			followingDay.append(y[0][i+1][0])


		# print str(pastFiveDays)
		# print str(followingDay)


		self.clf = GaussianNB()
		self.clf.fit(pastFiveDays, followingDay)



	'''
		Accepts arguments
		x: a single sequence of input of length similar to training data

		Outputs a list of digits between 0 and 1 for all timesteps in the sequence
	'''
	def predict(self, x):
		assert len(x) == self.max_length, "ERROR: " + str(len(x)) + " instead of " + str(self.max_length)
		assert len(x[0]) == self.num_inputs, "ERROR: " + str(len(x[0])) + " instead of " + str(self.num_inputs)


		last = len(x)
		# print str(last)
		# print str(x[last-6:last-1])
		# print str(np.reshape(x[last-6:last-1], 20))
		# print str(len(np.reshape(x[last-6:last-1], 20)))

		ret = self.clf.predict(np.reshape(x[last-6:last-1], 20))
		ret = ret.reshape(1, 1)

		return ret

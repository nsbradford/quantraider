import numpy as np
from sklearn import svm

import time

class SVM(object):
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

		self.num_inputs = len(x[0][0])
		self.num_outputs = len(y[0][0])
		self.max_length = len(x[0])
		train_x = []
		train_y = []

		print "input: " + str(x)
		print "output: " + str(y)

		for y1 in y[0]:
			train_y.append(y1[0])


		#self.clf = svm.SVR()
		#self.clf = svm.SVC()
		#self.clf = svm.NuSVC()#BAD
		#self.clf = svm.NuSVR()#BAD
		#self.clf = svm.OneClassSVM()
		#self.clf = svm.LinearSVC()
		self.clf = svm.SVC(C=10)
		self.clf.fit(x[0], train_y)



	'''
		Accepts arguments
		x: a single sequence of input of length similar to training data

		Outputs a list of digits between 0 and 1 for all timesteps in the sequence
	'''
	def predict(self, x):
		assert len(x) == self.max_length, "ERROR: " + str(len(x)) + " instead of " + str(self.max_length)
		assert len(x[0]) == self.num_inputs, "ERROR: " + str(len(x[0])) + " instead of " + str(self.num_inputs)
		#print len(x), self.max_length, len(x[0]), self.num_inputs
		#print "\n\n\nPREDICTING ON" + str(x)
		ret =  self.clf.predict(x)
		#print "PREDICTING " + str(ret)
		ret = ret.reshape((self.max_length, self.num_outputs))
		#print "PREDICTING " + str(ret)
		return ret

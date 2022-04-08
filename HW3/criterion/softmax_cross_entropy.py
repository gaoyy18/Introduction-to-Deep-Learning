""" Softmax Cross-Entropy Loss Layer """

import numpy as np

# a small number to prevent dividing by zero, maybe useful for you
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.acc = 0.
		self.loss = np.zeros(1, dtype='f')


	def forward(self, logit, gt):
		"""
	      Inputs: (minibatch)
	      - logit: forward results from the last FCLayer, shape(batch_size, 10)
	      - gt: the ground truth label, shape(batch_size, 10)
	    """

		############################################################################
	    # TODO: Put your code here
		# Calculate the average accuracy and loss over the minibatch, and
		# store in self.accu and self.loss respectively.
		# Only return the self.loss, self.accu will be used in solver.py.
		self.batch_size = 100
		self.logit = logit
		self.gt = gt
		hk = np.true_divide(np.exp(self.logit), np.tile(np.sum(np.exp(self.logit), axis=1), (10, 1)).T)
		self.loss = -np.sum(np.multiply(np.log(hk), gt)) / (self.batch_size * 10)
		self.acc = np.equal(np.argmax(self.logit, axis=1), np.argmax(self.gt, axis=1)).tolist().count(True) / self.batch_size


	    ############################################################################

		return self.loss


	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		return self.logit - self.gt

	    ############################################################################

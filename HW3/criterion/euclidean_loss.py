""" Euclidean Loss Layer """
import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.

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
		self.loss = np.sum(np.multiply((self.logit - self.gt), (self.logit - self.gt))) / (2 * self.batch_size * 10)
		self.acc = np.equal(np.argmax(self.logit, axis=1), np.argmax(self.gt, axis=1)).tolist().count(True) / self.batch_size


	    ############################################################################

		return self.loss

	def backward(self):

		############################################################################
	    # TODO: Put your code here
		# Calculate and return the gradient (have the same shape as logit)
		return self.logit - self.gt

	    ############################################################################

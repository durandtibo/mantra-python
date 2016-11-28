
import numpy as np

class PrimalModel:
	""" w: model parameters. w is an np.array """

	def __init__(self, w=None):
		self.w = w

	def get_dimension(self):
		""" Return the dimension of vector w or None if w is not an numpy.ndarray """
		if isinstance(self.w, np.ndarray):
			return self.w.size
		else:
			return None

	def norm2(self):
		return np.dot(self.w, self.w)

	def __str__(self):
		return 'PrimalModel [dimension=%s]' % (self.get_dimension())

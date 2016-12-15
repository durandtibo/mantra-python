import time
from abc import ABC, abstractmethod

import numpy as np
from mantra.util.primal_model import PrimalModel
from mantra.util.ranking import RankingLabel, RankingUtils
from mantra.util.ranking_cython import ssvm_ranking_feature_map


###############################################################################
# MantraModel
###############################################################################

class MantraModel(ABC, PrimalModel):

	def __init__(self, w=None, verbose=False):
		PrimalModel.__init__(self, w)
		self.verbose = verbose


	def __str__(self):
		return 'MantraModel [dimension=%d]' % self.get_dimension()


	@abstractmethod
	def loss(self, y_truth, y):
		""" Compute the loss value between y_truth and y: Delta(y_truth, y).
		- y_truth: ground truth label
		- y: label
		- return the loss value """
		pass


	@abstractmethod
	def max_oracle(self, x, y_star):
		pass


	@abstractmethod
	def predict(self, x, y):
		""" Predict the label of pattern x
		- x: pattern
		- y: label """
		pass


	@abstractmethod
	def feature_map(self, x, y, h):
		""" Compute the feature map psi(x, y, h).
		- x: pattern
		- y: label
		- h=(h^+,h^-): couple of latent variable
		- return vector psi(x, y, h) """
		pass


	def value_of(self, x, y=None, h=None):
		""" Compute the score of pattern x, label y and latent h
		- x: pattern
		- y: label
		- h=(h^+,h^-): couple of latent variable
		- return w dot psi(x, y, h)"""
		if y is None:
			y, h = self.predict(x)
		if h is None:
			h = self.predict(x, y)
		return self.dot(self.feature_map(x, y, h))


	def dot(self, psi):
		""" Compute the score of vector psi
		- psi: vector
		- return w dot psi"""
		return np.dot(self.w, psi)


	def add(self, vector, x, y, h, gamma=None):
		if gamma is None:
			vector += self.feature_map(x, y, h)
		else:
			vector += gamma * self.feature_map(x, y, h)
		return vector


	def sub(self, vector, x, y, h):
		vector -= self.feature_map(x, y, h)
		return vector


	@abstractmethod
	def initialization(self, data):
		""" Initialize model hyper-parameters and checks data """
		pass



###############################################################################
# MultiClassMantraModel4Bag
###############################################################################

class MultiClassMantraModel4Bag(MantraModel):

	def __init__(self, num_classes=None):
		MantraModel.__init__(self)
		if num_classes is None:
			self.num_classes = None
		else:
			self.num_classes = int(num_classes)


	def loss(self, y_truth, y):
		""" Compute the loss value between y_truth and y: Delta(y_truth, y).
		- y_truth: ground truth label
		- y: label
		- return the loss value """
		if y_truth == y:
			return 0.0
		else:
			return 1.0


	def feature_map(self, x, y, h):
		if y < 0 or y >= self.num_classes:
			raise ValueError('Error with label y (={}). The number of classes is {}'.format(y, self.num_classes))

		feature_dim = x.get_dimension()
		feature_vector = np.zeros(feature_dim * self.num_classes, dtype=np.float64)

		# Populate the feature_vector in blocks [<class-0 features> <class-1 features> ...].
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		instance = self.add_max_min_instances(x, h)
		feature_vector[start_idx:end_idx] += instance
		return feature_vector


	def max_oracle(self, x, y_star):
		num_instances = x.get_number_of_instances()
		w_multiclass = np.reshape(self.w, [self.num_classes, -1])
		scores = np.dot(w_multiclass, x.instances)
		indices = np.arange(self.num_classes)
		sort_idx = np.argsort(scores, axis=1)
		h_max = sort_idx[:,-1]
		h_min = sort_idx[:,0]
		scores_max = scores[indices, h_max]
		scores_min = scores[indices, h_min]
		scores_maxmin = scores_max + scores_min
		# add loss
		scores_maxmin += 1.0
		scores_maxmin[y_star] -= 1.0
		y_predict = np.argmax(scores_maxmin)
		return y_predict, [h_max[y_predict], h_min[y_predict]]


	def predict(self, x, y=None):
		if y is None:
			num_instances = x.get_number_of_instances()
			w_multiclass = np.reshape(self.w, [self.num_classes, -1])
			scores = np.dot(w_multiclass, x.instances)
			indices = np.arange(self.num_classes)
			sort_idx = np.argsort(scores, axis=1)
			h_max = sort_idx[:,-1]
			h_min = sort_idx[:,0]
			scores_max = scores[indices, h_max]
			scores_min = scores[indices, h_min]
			scores_maxmin = scores_max + scores_min
			y_predict = np.argmax(scores_maxmin)
			return y_predict, [h_max[y_predict], h_min[y_predict]]
		else:
			h_predict, score = self.predict_latent(x, y)
			return h_predict


	def predict_latent(self, x, y):
		# get the model for class y
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		wy = self.w[start_idx:end_idx]
		# compute the scores for each latent variable
		scores = np.dot(wy, x.instances)
		# find the label and the pair of latent variables with the maximum score
		sort_idx = np.argsort(scores)
		h_max = sort_idx[-1]
		h_min = sort_idx[0]
		h_predict = (h_max, h_min)
		score_max = scores[h_max]
		score_min = scores[h_min]
		score = score_max + score_min
		return h_predict, score


	def value_of(self, x, y=None, h=None):
		if y is None:
			y, h = self.predict(x)
		if h is None:
			h, score = self.predict_latent(x, y)
			#return score
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		wy = self.w[start_idx:end_idx]
		instance = self.add_max_min_instances(x, h)
		value = np.dot(wy, instance)
		return value


	def initialization(self, data):
		""" Initialize model hyper-parameters and checks data
			Warning: label is (label, (h^+, h^-)) to be compatible with the solver
		"""

		if self.num_classes is None:
			# initialize the number of classes
			self.num_classes = 0
			for example in data:
				self.num_classes = max(self.num_classes, example.label)
			self.num_classes += 1

		dimension = self.get_dimension()
		if dimension is None:
			if self.verbose:
				print('Initalize parameters with zeros')
			# use first example to determine the dimension of w
			feature_dimension = data[0].pattern.get_dimension()
			self.w = np.zeros(self.num_classes * feature_dimension)


	def compute_prediction_and_labels(self, data):
		prediction_and_labels = list()
		for example in data:
			label = example.label
			y_predict, h_predict = self.predict(example.pattern)
			prediction_and_labels.append([y_predict, label])
		return prediction_and_labels


	def __str__(self):
		return 'MultiClassMantraModel4Bag [num_classes={}, dimension={}]'.format(self.num_classes, self.get_dimension())


	def add(self, vector, x, y, h, gamma=None):
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		instance = self.add_max_min_instances(x, h)
		if gamma is None:
			vector[start_idx:end_idx] += instance
		else:
			vector[start_idx:end_idx] += gamma * instance
		return vector


	def sub(self, vector, x, y, h):
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		instance = self.add_max_min_instances(x, h)
		vector[start_idx:end_idx] -= instance
		return vector


	def add_max_min_instances(self, x, h):
		""" Returns the sum of instance h^+ and h^- """
		instance = x.get_instance(h[0]) + x.get_instance(h[1])
		return instance


	def get_all_scores(self, x, y):
		"""Compute the score of bag instances for class y.

		Args:
			x (Bag): The bag.
			y (int): The class.

		Returns:
			1d np.array: The score of bag instances for class y.

		"""
		# get the model for class y
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		wy = self.w[start_idx:end_idx]
		# compute the scores for each latent variable
		scores = np.dot(wy, x.instances)
		return scores



###############################################################################
# MultiClassMultiInstanceMantraModel4Bag
###############################################################################

class MultiClassMultiInstanceMantraModel4Bag(MantraModel):

	def __init__(self, k=1, num_classes=None):
		"""
		- k: number of selected instance
		"""
		MantraModel.__init__(self)
		self.k = k
		if num_classes is None:
			self.num_classes = None
		else:
			self.num_classes = int(num_classes)


	def loss(self, y_truth, y):
		""" Compute the loss value between y_truth and y: Delta(y_truth, y).
		- y_truth: ground truth label
		- y: label
		- return the loss value """
		if y_truth == y:
			return 0.0
		else:
			return 1.0


	def feature_map(self, x, y, h):
		if y < 0 or y >= self.num_classes:
			raise ValueError('Error with label y (={}). The number of classes is {}'.format(y, self.num_classes))

		feature_dim = x.get_dimension()
		feature_vector = np.zeros(feature_dim * self.num_classes, dtype=np.float64)

		# Populate the feature_vector in blocks [<class-0 features> <class-1 features> ...].
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		instance = self.add_max_min_instances(x, h)
		feature_vector[start_idx:end_idx] += instance
		return feature_vector


	def max_oracle(self, x, y_star):
		num_instances = x.get_number_of_instances()
		n = self.k # n is the number of selected instances
		if self.k < 1:
			n = max(1, self.k * num_instances)
		n = int(n)
		w_multiclass = np.reshape(self.w, [self.num_classes, -1])
		scores = np.dot(w_multiclass, x.instances)
		# sort scores for each row
		sort_idx = np.argsort(scores, axis=1)
		# predict latent
		h_max = sort_idx[:,-n:]
		h_min = sort_idx[:,:n]
		# compute score of max instances
		scores_max = scores[:,h_max]
		scores_max = np.sum(scores_max[:,0,:], axis=1)
		# compute score of min instances
		scores_min = scores[:,h_min]
		scores_min = np.sum(scores_min[:,0,:], axis=1)
		# score = max + min
		scores_maxmin = scores_max + scores_min
		# add loss score
		scores_maxmin += 1.0
		scores_maxmin[y_star] -= 1.0
		# predict label
		y_predict = np.argmax(scores_maxmin)
		return y_predict, [(h_max[y_predict])[::-1], h_min[y_predict]]


	def predict(self, x, y=None):
		if y is None:
			num_instances = x.get_number_of_instances()
			n = self.k # n is the number of selected instances
			if self.k < 1:
				n = max(1, self.k * num_instances)
			n = int(n)
			w_multiclass = np.reshape(self.w, [self.num_classes, -1])
			scores = np.dot(w_multiclass, x.instances)
			# sort scores for each row
			sort_idx = np.argsort(scores, axis=1)
			# predict latent
			h_max = sort_idx[:,-n:]
			h_min = sort_idx[:,:n]
			# compute score of max instances
			scores_max = scores[:,h_max]
			scores_max = np.sum(scores_max[:,0,:], axis=1)
			# compute score of min instances
			scores_min = scores[:,h_min]
			scores_min = np.sum(scores_min[:,0,:], axis=1)
			# score = max + min
			scores_maxmin = scores_max + scores_min
			# predict label
			y_predict = np.argmax(scores_maxmin)
			return y_predict, [(h_max[y_predict])[::-1], h_min[y_predict]]
		else:
			h_predict, score = self.predict_latent(x, y)
			return h_predict


	def predict_latent(self, x, y):
		num_instances = x.get_number_of_instances()
		n = self.k # n is the number of selected instances
		if self.k < 1:
			n = max(1, self.k * num_instances)
		n = int(n)
		# get the model for class y
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		wy = self.w[start_idx:end_idx]
		# compute the scores for each latent variable
		scores = np.dot(wy, x.instances)
		# find the label and the latent variable with the maximum score
		sort_idx = np.argsort(scores)
		h_max = sort_idx[-n:]
		h_min = sort_idx[:n]
		h_predict = [h_max[::-1], h_min]
		score = np.sum(scores[h_max]) + np.sum(scores[h_min])
		return h_predict, score


	def value_of(self, x, y=None, h=None):
		if y is None:
			y, h = self.predict(x)
		if h is None:
			h, score = self.predict_latent(x, y)
			#return score
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		wy = self.w[start_idx:end_idx]
		instance = self.add_max_min_instances(x, h)
		value = np.dot(wy, instance)
		return value


	def initialization(self, data):
		""" Initialize model hyper-parameters and checks data
			Warning: label is (label, (h^+, h^-)) to be compatible with the solver
		"""

		if self.num_classes is None:
			# initialize the number of classes
			self.num_classes = 0
			for example in data:
				self.num_classes = max(self.num_classes, example.label)
			self.num_classes += 1

		dimension = self.get_dimension()
		if dimension is None:
			if self.verbose:
				print('Initalize parameters with zeros')
			# use first example to determine the dimension of w
			feature_dimension = data[0].pattern.get_dimension()
			self.w = np.zeros(self.num_classes * feature_dimension)


	def compute_prediction_and_labels(self, data):
		prediction_and_labels = list()
		for example in data:
			label = example.label
			y_predict, h_predict = self.predict(example.pattern)
			prediction_and_labels.append([y_predict, label])
		return prediction_and_labels


	def __str__(self):
		return 'MultiClassMultiInstanceMantraModel4Bag [k={}, num_classes={}, dimension={}]'.format(self.k, self.num_classes, self.get_dimension())


	def add(self, vector, x, y, h, gamma=None):
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		instance = self.add_max_min_instances(x, h)
		if gamma is None:
			vector[start_idx:end_idx] += instance
		else:
			vector[start_idx:end_idx] += gamma * instance
		return vector


	def sub(self, vector, x, y, h):
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		instance = self.add_max_min_instances(x, h)
		vector[start_idx:end_idx] -= instance
		return vector

	def sum_instances(self, x, h):
		return np.sum(x.instances[:,h], axis=1)


	def add_max_min_instances(self, x, h):
		""" Returns the sum of instance h^+ and h^- """
		instance = self.sum_instances(x, h[0]) + self.sum_instances(x, h[1])
		return instance

	def get_all_scores(self, x, y):
		"""Compute the score of bag instances for class y.

		Args:
			x (Bag): The bag.
			y (int): The class.

		Returns:
			1d np.array: The score of bag instances for class y.

		"""
		# get the model for class y
		feature_dim = x.get_dimension()
		start_idx = y * feature_dim
		end_idx = start_idx + feature_dim
		wy = self.w[start_idx:end_idx]
		# compute the scores for each latent variable
		scores = np.dot(wy, x.instances)
		return scores


###############################################################################
# RankingAPMantraModel4Bag
###############################################################################

class RankingAPMantraModel4Bag(MantraModel):

	def __init__(self, num_classes=None):
		MantraModel.__init__(self)


	def loss(self, y_truth, y):
		""" Compute the loss value between y_truth and y: Delta(y_truth, y, h).
		- y_truth: ground truth label
		- y: label
		- return the loss value """
		return 1.0 - RankingUtils.average_precision(y_truth, y)


	def feature_map(self, x, y, h):
		number_of_examples = len(x.patterns)
		dimension = x.patterns[0].get_dimension()
		patterns = np.zeros([number_of_examples, dimension], dtype=np.float64)
		for i in range(number_of_examples):
			patterns[i] += x.patterns[i].get_instance(h[i][0])
			patterns[i] += x.patterns[i].get_instance(h[i][1])
		psi = ssvm_ranking_feature_map(patterns, x.labels, y.ranking)
		psi /= float(x.num_pos * x.num_neg)
		return psi


	def max_oracle(self, x, y_star):

		number_of_examples = len(x.patterns)
		positive_examples = list()
		negative_examples = list()

		h_predict = np.zeros([number_of_examples, 2], dtype=np.int32)

		for i in range(number_of_examples):
			h, score = self.predict_latent(x.patterns[i])
			h_predict[i] = h
			if y_star.labels[i] == 1:
				positive_examples.append((i, score))
			else:
				negative_examples.append((i, score))

		# sorts positive_examples and negative_examples in descending order of score
		positive_examples_sorted = sorted(positive_examples, key=lambda example: example[1], reverse=True)
		negative_examples_sorted = sorted(negative_examples, key=lambda example: example[1], reverse=True)

		#
		positive_id = 0
		negative_id = 0
		example_index_map = np.zeros(number_of_examples, dtype=np.uint32)
		for i in range(number_of_examples):
			if x.labels[i] == 1:
				example_index_map[positive_examples_sorted[positive_id][0]] = positive_id
				positive_id += 1
			else:
				example_index_map[negative_examples_sorted[negative_id][0]] = negative_id
				negative_id += 1

		num_pos = len(positive_examples_sorted)
		positive_example_score = np.zeros(num_pos, dtype=np.float64)
		for i in range(num_pos):
			positive_example_score[i] = positive_examples_sorted[i][1]

		num_neg = len(negative_examples_sorted)
		negative_example_score = np.zeros(num_neg, dtype=np.float64)
		for i in range(num_neg):
			negative_example_score[i] = negative_examples_sorted[i][1]

		y_predict = RankingUtils.find_optimum_neg_locations(x, positive_example_score, negative_example_score, example_index_map)
		return y_predict, h_predict



	def predict(self, x, y=None):
		number_of_examples = len(x.patterns)
		h_predict = np.zeros([number_of_examples, 2], dtype=np.int32)
		if y is None:
			# sort examples in descending order of score
			index_and_scores = list()
			for i in range(number_of_examples):
				h, score = self.predict_latent(x.patterns[i])
				h_predict[i] = h
				index_and_scores.append((i, score))
			index_and_scores_sorted = sorted(index_and_scores, key=lambda example: example[1], reverse=True)

			# compute ranking
			ranking = np.zeros(number_of_examples, dtype=np.uint32)
			for i in range(number_of_examples):
				ranking[index_and_scores_sorted[i][0]] = number_of_examples - i

			return RankingLabel(ranking, list(), 0, 0), h_predict
		else:
			for i in range(number_of_examples):
				h, score = self.predict_latent(x.patterns[i])
				h_predict[i] = h
			return h_predict


	def predict_latent(self, x):
		scores = np.dot(self.w, x.instances)
		sort_idx = np.argsort(scores)
		h_max = sort_idx[-1]
		h_min = sort_idx[0]
		h_predict = np.asarray((h_max, h_min), dtype=np.int32)
		score = scores[h_max] + scores[h_min]
		return h_predict, score


	def initialization(self, data):
		""" Initialize model hyper-parameters and checks data """
		dimension = self.get_dimension()
		if dimension is None:
			if self.verbose:
				print('Initalize parameters with zeros')
			# Use the first example to determine the dimension of w
			dimension = data[0].pattern.patterns[0].get_dimension()
			self.w = np.zeros(dimension)


	def compute_scores_and_labels(self, data):
		scores_and_labels = list()
		for example in data:
			label = example.label
			h, score = self.predict_latent(example.pattern)
			scores_and_labels.append((score, label))
		return scores_and_labels


	def __str__(self):
		return 'RankingAPMantraModel4Bag [dimension={}]'.format(self.get_dimension())


	def get_all_scores(self, x):
		"""Compute the score of bag instances.

		Args:
			x (Bag): The bag.

		Returns:
			1d np.array: The score of bag instances.

		"""
		# compute the scores for each latent variable
		scores = np.dot(self.w, x.instances)
		return scores

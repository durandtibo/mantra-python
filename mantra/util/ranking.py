
import numpy as np
from mantra.util.data.labeled_object import LabeledObject
from mantra.util.ranking_cython import (average_precision_cython,
                                         find_optimum_neg_locations_cython,
                                         generate_ranking_from_labels_cython)


###############################################################################
# RankingPattern
###############################################################################

class RankingPattern:
	""" label=1 -> relevant
		label=0 -> irrelevant """

	def __init__(self, patterns, labels, num_pos=None, num_neg=None):
		self.patterns = patterns
		self.labels = labels
		if num_pos is None or num_neg is None:
			self.num_pos = 0
			self.num_neg = 0
			for label in labels:
				if label == 1:
					self.num_pos += 1
				elif label == 0:
					self.num_neg += 1
				else:
					raise ValueError('incorrect label: %s (expected 1 or 0)' % label)
		else:
			self.num_pos = num_pos
			self.num_neg = num_neg


	def __str__(self):
		return 'RankingPattern [patterns={}, labels={}, num_pos={}, num_neg={}]'.format(len(self.patterns), len(self.labels), self.num_pos, self.num_neg)



###############################################################################
# RankingLabel
###############################################################################

class RankingLabel:

	def __init__(self, ranking=None, labels=None, num_pos=None, num_neg=None):
		self.ranking = ranking
		self.labels = labels
		self.num_pos = num_pos
		self.num_neg = num_neg


	def generate_ranking_label(self, labels):

		num_examples = labels.shape[0]

		self.ranking = np.zeros(num_examples, np.int32)
		self.labels = np.copy(labels)
		self.num_pos = 0
		self.num_neg = 0

		# Initializes labels
		for label in labels:
			if label == 1:
				self.num_pos += 1
			elif label == 0:
				self.num_neg += 1
			else:
				raise ValueError('incorrect label: %s (expected 1 or 0)' % label)

		self.ranking = generate_ranking_from_labels_cython(labels)

	def __str__(self):
		return 'RankingLabel [ranking={}, labels={}, num_pos={}, num_neg={}]'.format(len(self.ranking), len(self.labels), self.num_pos, self.num_neg)




###############################################################################
# RankingUtils
###############################################################################

class RankingUtils:

	def generate_ranking_example(data, target_label=None):
		patterns = list()
		labels = list()
		for example in data:
			patterns.append(example.pattern)
			label = example.label
			if target_label is not None:
				if label is target_label:
					label = 1
				else:
					label = 0
			labels.append(label)

		# Converts the list in np.array
		try:
			patterns = np.asarray(patterns, np.float64)
		except TypeError:
			print('patterns can not be converted to np.array')
			pass
		labels = np.asarray(labels, np.int32)
		# generates the ranking pattern
		ranking_pattern = RankingPattern(patterns, labels)
		# initalizes the ranking label
		ranking_label = RankingLabel()
		# generates a ranking with the labels
		ranking_label.generate_ranking_label(labels)
		# generates a list of LabeledObject with 1 example
		ranking_data = list()
		ranking_data.append(LabeledObject(ranking_pattern, ranking_label))
		return ranking_data


	def average_precision(y_truth, y_predict):
		""" Computes the average precision of 2 RankingLabel
		- y_truth: RankingLabel
		- y_predict: RankingLabel
		"""
		# converts the ranking in "score"
		scores = np.asarray(y_predict.ranking, dtype=np.float64)
		return average_precision_cython(y_truth.labels, scores)


	def average_precision_python(y_truth, y_predict):

		number_of_examples = y_truth.num_pos + y_truth.num_neg

		# Stores rank of all examples
		ranking = np.zeros(number_of_examples, dtype=np.int32)
		# Stores list of images sorted by rank. Higher rank to lower rank
		sorted_examples = np.zeros(number_of_examples, dtype=np.int32)

		# Converts rank matrix to rank list
		indexes = np.arange(number_of_examples)
		for i in indexes:
			ranking[i] = 1
			for j in indexes:
				if y_predict.ranking[i] > y_predict.ranking[j]:
					ranking[i] += 1
			sorted_examples[number_of_examples - ranking[i]] = i

		# Computes prec@i
		pos_count = 0.
		total_count = 0.
		precision_at_i = 0.
		for i in indexes:
			label = y_truth.labels[sorted_examples[i]]
			if label == 1:
				pos_count += 1
			total_count += 1
			if label == 1:
				precision_at_i += pos_count / total_count
		precision_at_i /= pos_count
		return precision_at_i

	def find_optimum_neg_locations(x, positive_example_score, negative_example_score, example_index_map):
		ranking = find_optimum_neg_locations_cython(x.num_pos, x.num_neg, x.labels, positive_example_score, negative_example_score, example_index_map)
		y_predict = RankingLabel(ranking=ranking, labels=list(), num_pos=x.num_pos, num_neg=x.num_neg)
		return y_predict

	def find_optimum_neg_locations_python(x, positive_example_score, negative_example_score, example_index_map):

		max_value = 0.0
		current_value = 0.0
		max_index = -1
		num_pos = x.num_pos
		num_neg = x.num_neg
		optimum_loc_neg_example = np.zeros(num_neg, dtype=np.uint32)

		# for every jth negative image
		for j in np.arange(1, num_neg+1):
			max_value = 0
			max_index = num_pos + 1
			# k is what we are maximising over. There would be one k_max for each negative image j
			current_value = 0
			for k in reversed(np.arange(1, num_pos+1)):
				current_value += (1.0 / num_pos) * ((j / (j + k)) - ((j - 1.0) / (j + k - 1.0))) - (2.0 / (num_pos * num_neg)) * (positive_example_score[k-1] - negative_example_score[j-1])
				if current_value > max_value:
					max_value = current_value
					max_index = k
				optimum_loc_neg_example[j-1] = max_index

		return RankingUtils.encode_ranking_python(x, positive_example_score, negative_example_score, example_index_map, optimum_loc_neg_example)


	def encode_ranking_python(x, positive_example_score, negative_example_score, example_index_map, optimum_loc_neg_example):

		labels = x.labels
		number_of_examples = len(x.patterns)
		ranking = np.zeros(number_of_examples, dtype=np.int32)

		for i in range(number_of_examples):
			for j in range(i+1, number_of_examples):
				if labels[i] == labels[j]:
					if labels[i] == 1:
						if positive_example_score[example_index_map[i]] > positive_example_score[example_index_map[j]]:
							ranking[i] += 1
							ranking[j] -= 1
						elif positive_example_score[example_index_map[j]] > positive_example_score[example_index_map[i]]:
							ranking[i] -= 1
							ranking[j] += 1
						else:
							if i < j:
								ranking[i] += 1
								ranking[j] -= 1
							else:
								ranking[i] -= 1
								ranking[j] += 1

					else:
						if negative_example_score[example_index_map[i]] > negative_example_score[example_index_map[j]]:
							ranking[i] += 1
							ranking[j] -= 1
						elif negative_example_score[example_index_map[j]] > negative_example_score[example_index_map[i]]:
							ranking[i] -= 1
							ranking[j] += 1
						else:
							if i < j:
								ranking[i] += 1
								ranking[j] -= 1
							else:
								ranking[i] -= 1
								ranking[j] += 1

				elif labels[i] == 1 and labels[j] == 0:
					i_prime = example_index_map[i] + 1
					j_prime = example_index_map[j] + 1
					oj_prime = optimum_loc_neg_example[j_prime-1]

					if (oj_prime - i_prime - 0.5) > 0:
						ranking[i] += 1
						ranking[j] -= 1
					else:
						ranking[i] -= 1
						ranking[j] += 1

				elif labels[i] == 0 and labels[j] == 1:
					i_prime = example_index_map[i] + 1
					j_prime = example_index_map[j] + 1
					oi_prime = optimum_loc_neg_example[i_prime - 1]

					if (j_prime - oi_prime + 0.5) > 0:
						ranking[i] += 1
						ranking[j] -= 1
					else:
						ranking[i] -= 1
						ranking[j] += 1

		return RankingLabel(ranking=ranking, labels=list(), num_pos=x.num_pos, num_neg=x.num_neg)

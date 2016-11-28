
from wsltibo.util.ranking_cython import average_precision_cython
import numpy as np

class Evaluation:

	def multiclass_accuracy(prediction_and_labels, display=True):
		num_corrects = 0
		for example in prediction_and_labels:
			prediction = example[0]
			label = example[1]
			if prediction == label:
				num_corrects += 1

		acc = float(num_corrects) / float(len(prediction_and_labels))
		if display:
			print("accuracy=%f \tnumber of examples=%d \t number of correct predictions=%d" % (acc, len(prediction_and_labels), num_corrects))
		return acc


	def average_precision(scores_and_labels, display=True):

		number_of_examples = len(scores_and_labels)

		scores = np.zeros(number_of_examples, dtype=np.float64)
		labels = np.zeros(number_of_examples, dtype=np.int32)
		for i in range(number_of_examples):
			scores[i] = scores_and_labels[i][0]
			labels[i] = scores_and_labels[i][1]

		precision_at_i = average_precision_cython(labels, scores)

		if display:
			print("average precision=%f" % precision_at_i)

		return precision_at_i


	def average_precision_python(scores_and_labels, display=True):

		number_of_examples = len(scores_and_labels)
		indexes = np.arange(number_of_examples)

		scores = np.zeros(number_of_examples)
		labels = np.zeros(number_of_examples)
		for i in indexes:
			scores[i] = scores_and_labels[i][0]
			labels[i] = scores_and_labels[i][1]


		# Stores rank of all images
		ranking = np.zeros(number_of_examples, dtype=np.uint32)
		# Stores the list of examples sorted by rank. Higher rank to lower rank
		sorted_examples = np.zeros(number_of_examples, dtype=np.uint32)

		# convert rank matrix to rank list
		for i in indexes:
			ranking[i] = 1
			for j in indexes:
				if scores[i] > scores[j]:
					ranking[i] += 1
			sorted_examples[number_of_examples - ranking[i]] = i

		# Computes prec@i
		pos_count = 0.
		total_count = 0.
		precision_at_i = 0.
		for i in indexes:
			label = labels[sorted_examples[i]]
			if label == 1:
				pos_count += 1
			total_count += 1
			if label == 1:
				precision_at_i += pos_count / total_count
		precision_at_i /= pos_count

		if display:
			print("average precision=%f" % precision_at_i)

		return precision_at_i

import csv
import profile
import time

import numpy as np
from mantra import (BagReader, Evaluation, LabeledObject, MantraWithSGD,
                    MantraWithSSG, MultiClassMantraModel4Bag,
                    MultiClassMultiInstanceMantraModel4Bag, Preprocessing, RankingAPMantraModel4Bag, RankingUtils)


#from sklearn.preprocessing import normalize


class DatasetDemo:

	def read_dataset_csv(filename, verbose=False):
		""" return the list of images and labels """
		list_data = list()
		print('read', filename)
		with open(filename, newline='') as f:
			reader = csv.reader(f)
			rownum = 0
			for row in reader:
				if rownum == 0:
					header = row
				else:
					name = row[0]
					label = int(row[1])
					example = LabeledObject(name, label)
					list_data.append(example)
				rownum += 1

		if verbose:
			print('read %d examples' % len(list_data))

		return list(list_data)


def demo_mantra_multiclass():

	print('\n**************************')
	print('* Demo MANTRA multiclass *')
	print('**************************')

	# path to the data
	path_data = "/Users/thibautdurand/Desktop/data/json/uiuc"
	filename_train = path_data + "/train.csv"
	filename_test = path_data + "/test.csv"

	# Read train data
	# Read image name and labels
	list_data = DatasetDemo.read_dataset_csv(filename_train, True)
	# Read bags
	data = BagReader.read_data_json(list_data, path_data, True)
	# Preprocess the data
	train = Preprocessing.normalize_bag(data)

	# Read test data
	list_data = DatasetDemo.read_dataset_csv(filename_test, True)
	data = BagReader.read_data_json(list_data, path_data, True)
	test = Preprocessing.normalize_bag(data)

	# Define model
	model = MultiClassMantraModel4Bag()
	# Define solver
	solver = MantraWithSGD(num_epochs=50, lambdaa=1e-4)
	# Learn model
	solver.optimize(model, train)

	# Evaluate performance on train data
	prediction_and_labels = model.compute_prediction_and_labels(train)
	Evaluation.multiclass_accuracy(prediction_and_labels)

	# Evaluate performance on test data
	prediction_and_labels = model.compute_prediction_and_labels(test)
	Evaluation.multiclass_accuracy(prediction_and_labels)


def demo_mantra_ranking():

	print('\n***********************')
	print('* Demo MANTRA ranking *')
	print('***********************')

	# path to the data
	path_data = "/Users/thibautdurand/Desktop/data/json/uiuc"
	filename_train = path_data + "/train.csv"
	filename_test = path_data + "/test.csv"

	# Read train data
	# Read image name and labels
	list_data = DatasetDemo.read_dataset_csv(filename_train, True)
	# Read bags
	data = BagReader.read_data_json(list_data, path_data, True)
	# Preprocess the data
	train = Preprocessing.normalize_bag(data)
	# Define the positive and negative examples
	for example in train:
		if example.label == 2:
			example.label = 1
		else:
			example.label = 0

	# Read test data
	# Read image name and labels
	list_data = DatasetDemo.read_dataset_csv(filename_test, True)
	# Read bags
	data = BagReader.read_data_json(list_data, path_data, True)
	# Preprocess the data
	test = Preprocessing.normalize_bag(data)
	# Define the positive and negative examples
	for example in test:
		if example.label == 2:
			example.label = 1
		else:
			example.label = 0

	# Generate ranking example
	train_rank = RankingUtils.generate_ranking_example(train)

	# Define model
	model = RankingAPMantraModel4Bag()
	# Define solver
	solver = MantraWithSGD(num_epochs=50)
	# Learn model
	solver.optimize(model, train_rank)

	# Evaluate performance on train data
	scores_and_labels = model.compute_scores_and_labels(train)
	Evaluation.average_precision(scores_and_labels)

	# Evaluate performance on test data
	scores_and_labels = model.compute_scores_and_labels(test)
	Evaluation.average_precision(scores_and_labels)



if __name__ == "__main__":
	demo_mantra_multiclass()
	demo_mantra_ranking()

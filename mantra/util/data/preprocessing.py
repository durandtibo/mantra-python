
import numpy as np
from wsltibo.util.data.labeled_object import LabeledObject


class Preprocessing:

	def normalize(v):
		norm = np.linalg.norm(v)
		if norm == 0:
			return v
		return v / norm


	def normalize_bag(data, add_bias=True):
		for example in data:
			bag = example.pattern
			instances = bag.instances
			normalized_instances = list()
			for i in range(bag.get_number_of_instances()):
				normalized_instance = Preprocessing.normalize(instances[:,i])
				if add_bias:
					normalized_instance = np.concatenate((normalized_instance, np.array([1])), axis=0)
				normalized_instances.append(normalized_instance)
			bag.instances = np.transpose(np.asarray(normalized_instances, dtype=np.float64))

		return data


	def normalize_features(data, add_bias=True):
		for example in data:
			# feature = example.pattern.reshape(-1, 1)
			# example.pattern = normalize(feature, axis=0).flatten()
			example.pattern = Preprocessing.normalize(example.pattern)
			if add_bias:
				example.pattern = np.concatenate((example.pattern, np.array([1])), axis=0)

		return data


	def generate_super_examples_from_bags(data):
		list_examples = list()
		for example in data:
			bag = example.pattern
			label = example.label
			instances = bag.instances
			super_instance = None
			for i in range(bag.get_number_of_instances()):
				if super_instance is None:
					super_instance = bag.get_instance(i)
				else:
					super_instance += bag.get_instance(i)

			super_instance /= bag.get_number_of_instances()
			new_example = LabeledObject(super_instance, label)
			list_examples.append(new_example)

		return list_examples


	def generate_examples_from_bags(data):
		list_examples = list()
		for example in data:
			bag = example.pattern
			label = example.label
			for i in range(bag.get_number_of_instances()):
				new_example = LabeledObject(bag.get_instance(i), label)
				list_examples.append(new_example)

		return list_examples

	def binarize_dataset(data, target_label=None, positive_label=1, negative_label=-1):
		for example in data:
			if example.label is target_label:
				example.label = positive_label
			else:
				example.label = negative_label
		return data

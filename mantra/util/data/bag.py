
import json

import numpy as np
from mantra.util.data.labeled_object import LabeledObject
from mantra.util.progress_bar import ProgressBar


class Bag:
	"""" Bag is a class to manipulate bag structure
	- name: name of the bag
	- instances: list of instances """

	def __init__(self, name=None, instances=None):
		self.name = name
		self.instances = instances


	def get_instance(self, index):
		return self.instances[:,index]


	def get_number_of_instances(self):
		""" Return the number of instances in the bag. """
		if self.instances is None:
			return 0
		return self.instances.shape[1]


	def get_dimension(self):
		""" Return the dimension of the instances. """
		if self.instances is None:
			return 0
		return self.instances.shape[0]


	def __str__(self):
		return "Bag [name={}, number of instances={}]".format(self.name, self.get_number_of_instances())



class BagReader:

	def read_bag_json(filename):
		with open(filename) as json_data:
			json_data = open(filename)
			data = json.load(json_data)

			# read general information about the bag and create it
			name = data['name']
			num_instances = int(data['numberOfInstances'])
			instances = None

			# read feature of each instance
			for i in range(num_instances):
				values = data['instances'][i]['feature']
				feature = np.asarray(values, dtype=np.float64)
				if instances is None:
					instances = np.zeros([num_instances, feature.shape[0]], dtype=np.float64)
				instances[i] = feature

			bag = Bag(name, np.transpose(instances))

		return bag

	def read_data_json(list_data, path_data, verbose=False):
		""" return the list of bags and labels """
		data = list()
		number_of_instances = 0
		pb = ProgressBar(len(list_data), 'Reading bags')
		pb.start()
		for example in list_data:
			pb.step()
			name = example.pattern
			label = example.label
			filename = "{}/{}.json".format(path_data, name)
			bag = BagReader.read_bag_json(filename)
			number_of_instances += bag.get_number_of_instances()
			data.append(LabeledObject(bag, label))
		pb.stop()

		avg_number_of_instances = number_of_instances / len(list_data)

		if verbose:
			print("Read {} bags with {} instances per bag ".format(len(list_data), avg_number_of_instances))

		return data

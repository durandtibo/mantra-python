
import numpy as np
from wsltibo.util.data.bag.bag import Bag


def test_bag_get_instance():
	instances = np.reshape(np.arange(20), [4, 5])
	bag = Bag("A", instances)
	# test instance 0
	instance = bag.get_instance(0)
	assert np.array_equal(instance, [0, 5, 10, 15])
	# test instance 3
	instance = bag.get_instance(3)
	assert np.array_equal(instance, [3, 8, 13, 18])


def test_bag_get_number_of_instances():
	# test with 5 instances of dimension 4
	instances = np.reshape(np.arange(20), [4, 5])
	bag = Bag("A", instances)
	assert bag.get_number_of_instances() == 5

	# test with 10 instances of dimension 5
	instances = np.reshape(np.arange(50), [5, 10])
	bag = Bag("B", instances)
	assert bag.get_number_of_instances() == 10

	# test with no instance
	bag = Bag("C")
	assert bag.get_number_of_instances() == 0


def test_bag_get_dimension():
	# test with 5 instances of dimension 4
	instances = np.reshape(np.arange(20), [4, 5])
	bag = Bag("A", instances)
	assert bag.get_dimension() == 4

	# test with 10 instances of dimension 5
	instances = np.reshape(np.arange(50), [5, 10])
	bag = Bag("B", instances)
	assert bag.get_dimension() == 5

	# test with no instance
	bag = Bag("C")
	assert bag.get_dimension() == 0

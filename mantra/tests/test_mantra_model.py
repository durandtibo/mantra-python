
import numpy as np
import pytest

from mantra.mantra_model import (MultiClassMantraModel4Bag,
                                 MultiClassMultiInstanceMantraModel4Bag)
from mantra.util.data.bag import Bag


###############################################################################
# MultiClassMantraModel4Bag
###############################################################################

def initialize_multiclass_model():
	# define feature dimension
	dimension = 5
	# initialize model with 4 classes
	model = MultiClassMantraModel4Bag(4)
	model.w = np.arange(model.num_classes * dimension, dtype=np.float64)
	return model, dimension

def initialize_bag(dimension, num_instances=7):
	instances = np.reshape(np.arange(dimension * num_instances), [dimension, num_instances])
	bag = Bag("A", instances)
	return bag


def test_multiclass_loss():
	model, dimension = initialize_multiclass_model()
	assert model.loss(0, 0) == 0
	assert model.loss(1, 0) == 1
	assert model.loss(0, 1) == 1
	assert model.loss(1, 1) == 0


def test_multiclass_add_max_min_instances():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)
	h = (1, 3)

	instance = model.add_max_min_instances(bag, h)
	instance_true = bag.get_instance(h[0]) + bag.get_instance(h[1])
	assert np.array_equal(instance, instance_true)


def test_multiclass_feature_map():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	h = (1, 3)
	feature_map = model.feature_map(bag, 0, h)
	feature_map_true = np.zeros(model.num_classes * dimension)
	instance = model.add_max_min_instances(bag, h)
	feature_map_true[0:dimension] += instance
	assert np.array_equal(feature_map, feature_map_true)

	h = (0, 4)
	feature_map = model.feature_map(bag, 2, h)
	feature_map_true = np.zeros(model.num_classes * dimension)
	instance = model.add_max_min_instances(bag, h)
	feature_map_true[2*dimension:3*dimension] += instance
	assert np.array_equal(feature_map, feature_map_true)


def test_multiclass_max_oracle():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	y_star, h_star = model.max_oracle(bag, 0)
	assert y_star == 3 # label
	assert h_star[0] == 6	# h^+
	assert h_star[1] == 0	# h^-

	model.w = np.ones(model.num_classes * dimension, dtype=np.float64)
	y_star, h_star = model.max_oracle(bag, 0)
	assert y_star == 1 # label
	assert h_star[0] == 6	# h^+
	assert h_star[1] == 0	# h^-

	y_star, h_star = model.max_oracle(bag, 1)
	assert y_star == 0 # label
	assert h_star[0] == 6	# h^+
	assert h_star[1] == 0	# h^-


def test_multiclass_predict():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	y_star, h_star = model.predict(bag)
	assert y_star == 3 # label
	assert h_star[0] == 6	# h^+
	assert h_star[1] == 0	# h^-

	h_star = model.predict(bag, 0)
	assert h_star[0] == 6	# h^+
	assert h_star[1] == 0	# h^-

	model.w = -model.w
	y_star, h_star = model.predict(bag)
	assert y_star == 0 # label
	assert h_star[0] == 0	# h^+
	assert h_star[1] == 6	# h^-

	h_star = model.predict(bag, 1)
	assert h_star[0] == 0	# h^+
	assert h_star[1] == 6	# h^-


def test_multiclass_predict_latent():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	h_star, score = model.predict_latent(bag, 0)
	assert h_star[0] == 6	# h^+
	assert h_star[1] == 0	# h^-

	model.w = -model.w
	h_star, score = model.predict_latent(bag, 1)
	assert h_star[0] == 0	# h^+
	assert h_star[1] == 6	# h^-


def test_multiclass_value_of():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	model.w = np.ones(model.num_classes * dimension, dtype=np.float64)
	score = model.value_of(bag, 0, (0, 6))
	score_true = np.sum(model.add_max_min_instances(bag, (0, 6)))
	assert score == 170

	model.w = np.zeros(model.num_classes * dimension, dtype=np.float64)
	score = model.value_of(bag, 0, (0, 6))
	assert score == 0


def test_multiclass_add():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	vector = np.zeros(model.num_classes * dimension, dtype=np.float64)
	model.add(vector, bag, 0, (0, 6))
	vector_true = [6, 20, 34, 48, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	assert np.array_equal(vector, vector_true)

	model.add(vector, bag, 3, (0, 6))
	vector_true = [6, 20, 34, 48, 62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 20, 34, 48, 62]
	assert np.array_equal(vector, vector_true)

	model.add(vector, bag, 2, (0, 6), 0.5)
	vector_true = [6, 20, 34, 48, 62, 0, 0, 0, 0, 0, 3, 10, 17, 24, 31, 6, 20, 34, 48, 62]
	assert np.array_equal(vector, vector_true)

	model.add(vector, bag, 1, (0, 6), -0.5)
	vector_true = [6, 20, 34, 48, 62, -3, -10, -17, -24, -31, 3, 10, 17, 24, 31, 6, 20, 34, 48, 62]
	assert np.array_equal(vector, vector_true)


def test_multiclass_sub():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	vector = np.zeros(model.num_classes * dimension, dtype=np.float64)
	model.sub(vector, bag, 0, (0, 6))
	vector_true = [-6, -20, -34, -48, -62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	assert np.array_equal(vector, vector_true)

	model.sub(vector, bag, 3, (0, 6))
	vector_true = [-6, -20, -34, -48, -62, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6, -20, -34, -48, -62]
	assert np.array_equal(vector, vector_true)


def test_multiclass_get_all_scores():
	model, dimension = initialize_multiclass_model()
	bag = initialize_bag(dimension)

	scores = model.get_all_scores(bag, 0)
	assert np.array_equal(scores, [210, 220, 230, 240, 250, 260, 270])



###############################################################################
# MultiClassMultiInstanceMantraModel4Bag
###############################################################################

def initialize_multiclass_multi_instance_model():
	# define feature dimension
	dimension = 2
	# initialize model with 1 instance and 4 classes
	model = MultiClassMultiInstanceMantraModel4Bag(1, 4)
	model.w = np.arange(model.num_classes * dimension, dtype=np.float64)
	return model, dimension


def initialize_bag2(dimension, num_instances=10):
	instances = np.reshape(np.arange(dimension * num_instances), [dimension, num_instances])
	bag = Bag("B", instances)
	return bag


def test_multiclass_multi_instance_loss():
	model, dimension = initialize_multiclass_multi_instance_model()
	assert model.loss(0, 0) == 0
	assert model.loss(1, 0) == 1
	assert model.loss(0, 1) == 1
	assert model.loss(1, 1) == 0


def test_multiclass_multi_instance_sum_instances():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	h = [1, 3]
	instance = model.sum_instances(bag, h)
	instance_true = [4, 24]
	assert np.array_equal(instance, instance_true)

	h = [0, 4, 9]
	instance = model.sum_instances(bag, h)
	instance_true = [13, 43]
	assert np.array_equal(instance, instance_true)


def test_multiclass_multi_instance_add_max_min_instances():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	h = [[1], [3]]
	instance = model.add_max_min_instances(bag, h)
	instance_true = [4, 24]
	assert np.array_equal(instance, instance_true)

	h = [[0, 7], [4, 9]]
	instance = model.add_max_min_instances(bag, h)
	instance_true = [20, 60]
	assert np.array_equal(instance, instance_true)


def test_multiclass_multi_instance_feature_map():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	h = [[1], [3]]
	feature_vector = model.feature_map(bag, 0, h)
	feature_vector_true = [4, 24, 0, 0, 0, 0, 0, 0]
	assert np.array_equal(feature_vector, feature_vector_true)

	h = [[0, 7], [4, 9]]
	feature_vector = model.feature_map(bag, 3, h)
	feature_vector_true = [0, 0, 0, 0, 0, 0, 20, 60]
	assert np.array_equal(feature_vector, feature_vector_true)


def test_multiclass_multi_instance_max_oracle():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	y_star, h_star = model.max_oracle(bag, 0)
	assert y_star == 3 # label
	assert h_star[0] == [9]	# h^+
	assert h_star[1] == [0]	# h^-

	model.k = 3
	y_star, h_star = model.max_oracle(bag, 0)
	assert y_star == 3#3 # label
	assert np.array_equal(h_star[0], [9, 8, 7])	# h^+
	assert np.array_equal(h_star[1], [0, 1, 2])	# h^-

	model.k = 0.2
	y_star, h_star = model.max_oracle(bag, 0)
	assert y_star == 3#3 # label
	assert np.array_equal(h_star[0], [9, 8])	# h^+
	assert np.array_equal(h_star[1], [0, 1])	# h^-

	model.w = np.ones(model.num_classes * dimension, dtype=np.float64)
	y_star, h_star = model.max_oracle(bag, 0)
	assert y_star == 1 # label
	assert np.array_equal(h_star[0], [9, 8])	# h^+
	assert np.array_equal(h_star[1], [0, 1])	# h^-

	y_star, h_star = model.max_oracle(bag, 1)
	assert y_star == 0 # label
	assert np.array_equal(h_star[0], [9, 8])	# h^+
	assert np.array_equal(h_star[1], [0, 1])	# h^-

def test_multiclass_multi_instance_predict():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	# model.k = 2
	y_star, h_star = model.predict(bag)
	assert y_star == 3#3 # label
	assert h_star[0] == [9]	# h^+
	assert h_star[1] == [0]	# h^-

	model.k = 3
	y_star, h_star = model.predict(bag)
	assert y_star == 3#3 # label
	assert np.array_equal(h_star[0], [9, 8, 7])	# h^+
	assert np.array_equal(h_star[1], [0, 1, 2])	# h^-

	model.k = 0.2
	y_star, h_star = model.predict(bag)
	assert y_star == 3#3 # label
	assert np.array_equal(h_star[0], [9, 8])	# h^+
	assert np.array_equal(h_star[1], [0, 1])	# h^-

	model.w = np.ones(model.num_classes * dimension, dtype=np.float64)
	h_star = model.predict(bag, 0)
	assert np.array_equal(h_star[0], [9, 8])	# h^+
	assert np.array_equal(h_star[1], [0, 1])	# h^-

	h_star = model.predict(bag, 3)
	assert np.array_equal(h_star[0], [9, 8])	# h^+
	assert np.array_equal(h_star[1], [0, 1])	# h^-


def test_multiclass_multi_instance_value_of():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	model.w = np.ones(model.num_classes * dimension, dtype=np.float64)
	h = [[1], [3]]
	score = model.value_of(bag, 0, h)
	assert score == 28

	h = [[0, 7], [4, 9]]
	score = model.value_of(bag, 2, h)
	assert score == 80

	model.w = np.zeros(model.num_classes * dimension, dtype=np.float64)
	h = [[1], [3]]
	score = model.value_of(bag, 1, h)
	assert score == 0

	h = [[0, 7], [4, 9]]
	score = model.value_of(bag, 3, h)
	assert score == 0


def test_multiclass_multi_instance_add():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	vector = np.zeros(model.num_classes * dimension, dtype=np.float64)
	h = [[1], [3]]
	model.add(vector, bag, 0, h)
	vector_true = [4, 24, 0, 0, 0, 0, 0, 0]
	assert np.array_equal(vector, vector_true)

	h = [[0, 7], [4, 9]]
	model.add(vector, bag, 3, h)
	vector_true = [4, 24, 0, 0, 0, 0, 20, 60]
	assert np.array_equal(vector, vector_true)

	h = [[1], [3]]
	model.add(vector, bag, 2, h, 0.5)
	vector_true = [4, 24, 0, 0, 2, 12, 20, 60]
	assert np.array_equal(vector, vector_true)

	h = [[0, 7], [4, 9]]
	model.add(vector, bag, 1, h, -0.5)
	vector_true = [4, 24, -10, -30, 2, 12, 20, 60]
	assert np.array_equal(vector, vector_true)


def test_multiclass_multi_instance_sub():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	vector = np.zeros(model.num_classes * dimension, dtype=np.float64)
	h = [[1], [3]]
	model.sub(vector, bag, 0, h)
	vector_true = [-4, -24, 0, 0, 0, 0, 0, 0]
	assert np.array_equal(vector, vector_true)

	h = [[0, 7], [4, 9]]
	model.sub(vector, bag, 3, h)
	vector_true = [-4, -24, 0, 0, 0, 0, -20, -60]
	assert np.array_equal(vector, vector_true)


def test_multiclass_multi_instance_get_all_scores():
	model, dimension = initialize_multiclass_multi_instance_model()
	bag = initialize_bag2(dimension)

	scores = model.get_all_scores(bag, 0)
	assert np.array_equal(scores, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

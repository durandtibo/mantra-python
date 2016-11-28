
import time

from mantra.util.data.labeled_object import LabeledObject
from mantra.util.solver.loss import Loss
from mantra.util.solver.sgd_solver import SGDSolver
from mantra.util.solver.ssg_solver import SSGSolver


class MantraLoss(Loss):

	def __str__(self):
		return "MantraLoss"


	def evaluate(self, model, x, y):
		""" Evaluate the loss for the given model, pattern and label. The return value is a scalar.
		- model: model
		- x: pattern
		- y: label """

		# compute the loss augmented inference
		y_star, h_star = self.max_oracle(model, x, y)

		# compute the best latent value for label y
		h_bar = model.predict(x, y)

		# compute the loss term
		return model.loss(y, y_star) + model.value_of(x, y_star, h_star) - model.value_of(x, y, h_bar)


	def compute_gradient(self, model, x, y, y_star2=None):
		""" Compute the gradient of the hinge loss for the given model, pattern and label. The return value is a vector.
		- model: model
		- x: pattern
		- y: label """

		if y_star2 is None:
			# compute the loss augmented inference
			y_star, h_star = self.max_oracle(model, x, y)
		else:
			y_star, h_star = y_star2

		if y_star == y:
			# for this case, the gradient is zero
			return None

		# compute the best latent value for label y
		h_bar = model.predict(x, y)

		# compute the gradient of the loss
		return model.sub(model.feature_map(x, y_star, h_star), x, y, h_bar)

	def error(self, model, y_truth, y):
		""" Compute the loss function
		- y_truth: label
		- y: label """
		return model.loss(y_truth, y)


	def max_oracle(self, model, x, y_star):
		""" Compute the loss-augmented inference defined in model for pattern x and label y
		- x: pattern
		- y_star: label
		return (label, (h^+, h^-)) """
		return model.max_oracle(x, y_star)


def initialize_mantra_data(model, data):
	initialized_data = list()
	for i in range(len(data)):
		pattern = data[i].pattern
		label = data[i].label
		latent = model.initialize_latent(pattern, label)
		initialized_data.append(LabeledObject(pattern, (label, latent)))
	return initialized_data


###############################################################################
# MantraWithSGD
###############################################################################

class MantraWithSGD:

	def __init__(self, lambdaa=1e-4, num_epochs=25, sample='perm', seed=1, verbose=1, show_debug_every=0):
		self.lambdaa = lambdaa
		self.num_epochs = num_epochs
		self.sample = sample
		self.seed = seed
		self.verbose = verbose
		self.show_debug_every = int(show_debug_every)


	def optimize(self, model, data, val_data=None):
		solver = SGDSolver(MantraLoss(), self.lambdaa, self.num_epochs, self.sample, self.seed, self.verbose, self.show_debug_every)
		return solver.optimize(model, data, val_data)



###############################################################################
# MantraWithSSG
###############################################################################

class MantraWithSSG:

	def __init__(self, lambdaa=1e-4, num_epochs=25, sample='perm', do_weighted_averaging=False, seed=1, verbose=1, show_debug_every=0):
		self.lambdaa = lambdaa
		self.num_epochs = num_epochs
		self.sample = sample
		self.do_weighted_averaging = do_weighted_averaging
		self.seed = seed
		self.verbose = verbose
		self.show_debug_every = int(show_debug_every)


	def optimize(self, model, data, val_data=None):
		solver = SSGSolver(MantraLoss(), self.lambdaa, self.num_epochs, self.sample, self.do_weighted_averaging, self.seed, self.verbose, self.show_debug_every)
		return solver.optimize(model, data, val_data)

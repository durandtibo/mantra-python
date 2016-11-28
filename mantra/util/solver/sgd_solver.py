
import time

import numpy as np
from mantra.util.data.labeled_object import LabeledObject
from mantra.util.progress_bar import ProgressBar
from mantra.util.solver.loss import HingeLoss
from mantra.util.solver.solver_utils import SolverUtils


class SGDSolver:

	def __init__(self, loss=None, lambdaa=1e-4, num_epochs=25, sample='perm', seed=1, verbose=1, show_debug_every=0):
		self.loss = loss
		self.lambdaa = lambdaa
		self.num_epochs = num_epochs
		self.sample = sample
		self.seed = seed
		self.verbose = verbose
		self.show_debug_every = int(show_debug_every)


	def optimize(self, model, data, val_data=None):

		if self.verbose > 0:
			print('-'*100)
			print(self.__str__())
			print('-'*100)

		# Get the number of examples
		num_examples = len(data)

		if self.show_debug_every == 0:
			self.show_debug_every = num_examples

		# Initialize model and checks data
		model.initialization(data)

		if self.verbose > 0:
			print('# loss=%s' % self.loss)
			print('# model=%s' % model)
			print('-'*100)

		# Get the feature dimension
		dimension = model.get_dimension()

		# Initialize random generator of numpy
		np.random.seed(self.seed)

		if self.verbose > 0:
			print('Training with %s examples of dimension %s' % (num_examples, dimension))

		# Define the array of indices
		indices = np.arange(num_examples)

		# Initialize the progress bar
		pb = None
		if self.verbose == 1:
			pb = ProgressBar(self.num_epochs, "Optimization", 50)
			pb.start()

		t = 1.0

		t_start_solver = self.get_time()

		for epoch in range(self.num_epochs):

			if self.sample == 'perm':
				np.random.shuffle(indices) # Shuffle the indices

			for index in indices:
				# Compute the learning rate for iteration t
				learning_rate = 1.0 / (1.0 + self.lambdaa * t)

				# Pick example i
				i = index
				example = data[i]
				pattern = example.pattern
				label = example.label

				# Get the (sub)gradient
				gradient = self.loss.compute_gradient(model, pattern, label)

				# Update w
				self.update_w(model, gradient, learning_rate)

				t += 1.0

				if self.verbose > 1 and int(t) % self.show_debug_every == 0:
					primal, norm_w2, loss_value = SolverUtils.primal_objective(model, data, self.loss, self.lambdaa)
					print('epoch={epoch} \titeration={t} \tprimal objective={primal} \t||w||^2={norm_w2} \tloss={loss_value}'.format(epoch=epoch, t=t, primal=primal, norm_w2=norm_w2, loss_value=loss_value))

			if self.verbose == 1:
				pb.step()	# Increment the progress bar

		# Get the time at the end of the optimization
		t_end_solver = self.get_time()
		time_solver = t_end_solver - t_start_solver

		if self.verbose == 1:
			pb.stop()	# Stop the progress bar

		if self.verbose > 0:
			primal, norm_w2, loss_value = SolverUtils.primal_objective(model, data, self.loss, self.lambdaa)
			print('* Optimization solved in {time}s \tprimal objective={primal} \t||w||^2={norm_w2} \tloss={loss_value}'.format(time=time_solver, epoch=epoch, t=t, primal=primal, norm_w2=norm_w2, loss_value=loss_value))
			print('-'*100)

		return model


	def __str__(self):
		string = '# Stochastic Gradient Descent options\n'
		string += '# lambda=%s\n' % self.lambdaa
		string += '# number of epochs=%d\n' % self.num_epochs
		string += '# sample=%s' % self.sample
		return string


	def get_time(self):
		""" Compute the current time in us"""
		return time.time()


	def update_w(self, model, gradient, learning_rate):
		# Update w (regularization term)
		model.w *= (1.0 - self.lambdaa * learning_rate)

		# Update w with the gradient if the gradient is not zero
		if gradient is not None:
			# Update w (loss term)
			gradient *= -learning_rate
			model.w += gradient

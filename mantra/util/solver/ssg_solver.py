
import time

import numpy as np
from mantra.util.data.labeled_object import LabeledObject
from mantra.util.progress_bar import ProgressBar
from mantra.util.solver.solver_utils import SolverUtils


class SSGSolver:

	def __init__(self, loss=None, lambdaa=1e-4, num_epochs=25, sample='perm', do_weighted_averaging=False, seed=1, verbose=1, show_debug_every=0):
		self.loss = loss
		self.lambdaa = lambdaa
		self.num_epochs = num_epochs
		self.sample = sample
		self.do_weighted_averaging = do_weighted_averaging
		self.seed = seed
		self.verbose = verbose
		self.show_debug_every = int(show_debug_every)


	def optimize(self, model, data, val_data=None):

		if self.verbose > 0:
			print('-'*100)
			print(self.__str__())
			print('-'*100)

		# Gets the number of examples
		num_examples = len(data)

		if self.show_debug_every == 0:
			self.show_debug_every = num_examples

		# Initialzes model and checks data
		model.initialization(data)

		# Gets the feature dimension
		dimension = model.get_dimension()

		# Initializes weighted averaging model
		if self.do_weighted_averaging:
			w_avg = np.zeros(dimension)

		# Initializes random generator
		np.random.seed(self.seed)

		if self.verbose > 0:
			print('Training with %s examples of dimension %s' % (num_examples, dimension))

		# Defines the indices
		indices = np.arange(num_examples)

		# Initializes the progress bar
		pb = None
		if self.verbose == 1:
			pb = ProgressBar(self.num_epochs, "Optimization", 50)
			pb.start()

		t = 1.0

		t_start_solver = self.get_time()

		for epoch in range(self.num_epochs):

			if self.sample == 'perm':
				np.random.shuffle(indices) # shuffles the indices

			for index in indices:
				# Computes the learning rate for iteration t
				learning_rate = 1.0 / (1.0 + self.lambdaa * t)

				# Picks example i
				i = index
				example = data[i]
				pattern = example.pattern
				label = example.label

				# Solves the loss-augmented inference for point i
				y_star = self.loss.max_oracle(model, pattern, label)

				# Step size gamma
				gamma = 1.0 / (t + 1.0)

				# Gets the (sub)gradient
				gradient = self.loss.compute_gradient(model, pattern, label, y_star)

				# Updates w
				self.update_w(model, gradient, gamma)

				if self.do_weighted_averaging:
					rho = 2.0 / (t + 2.0)
					model.w *= rho
					w_avg *= (1.0 - rho)
					w_avg += model.w
					model.w *= 1.0 / rho

				t += 1.0

				if self.verbose > 1 and int(t) % self.show_debug_every == 0:
					primal, norm_w2, loss_value = SolverUtils.primal_objective(model, data, self.loss, self.lambdaa)
					print('epoch={epoch} \titeration={t} \tprimal objective={primal} \t||w||^2={norm_w2} \tloss={loss_value}'.format(epoch=epoch, t=t, primal=primal, norm_w2=norm_w2, loss_value=loss_value))

			if self.verbose == 1:
				pb.step()	# increment the progress bar

		# gets the time at the end of the optimization
		t_end_solver = self.get_time()
		time_solver = t_end_solver - t_start_solver

		if self.verbose == 1:
			pb.stop()	# stops the progress bar

		if self.do_weighted_averaging:
			model.w = w_avg

		if self.verbose > 0:
			primal, norm_w2, loss_value = SolverUtils.primal_objective(model, data, self.loss, self.lambdaa)
			print('* Optimization solved in {time}s \tprimal objective={primal} \t||w||^2={norm_w2} \tloss={loss_value}'.format(time=time_solver, epoch=epoch, t=t, primal=primal, norm_w2=norm_w2, loss_value=loss_value))

		return model


	def update_w(self, model, gradient, gamma):

		# Update the weights of the model
		model.w *= (1.0 - gamma)

		# update w with the gradient if the gradient is not zero
		if gradient is not None:
			gradient *= (-gamma / self.lambdaa)
			model.w += gradient


	def __str__(self):
		string = '# Stochastic (Sub)Gradient Descent options\n'
		string += '# lambda=%s\n' % self.lambdaa
		string += '# number of epochs=%d\n' % self.num_epochs
		string += '# sample=%s\n' % self.sample
		string += '# do weighted averaging=%s\n' % self.do_weighted_averaging
		string += '# loss=%s' % self.loss
		return string


	def get_time(self):
		""" Compute the current time in us"""
		return time.time()

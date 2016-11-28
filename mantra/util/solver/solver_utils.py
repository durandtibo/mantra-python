
import numpy as np

class SolverUtils:


	def loss_function(model, data, loss):

		# Gets the number of examples
		num_examples = len(data)

		# Computes the loss value of each example
		loss_value = 0.0
		for i in range(num_examples):
			example = data[i]
			loss_value += loss.evaluate(model, example.pattern, example.label)

		return loss_value / num_examples


	def primal_objective(model, data, loss, lambdaa, c=1.0):
		# Compute the loss term
		loss_value = SolverUtils.loss_function(model, data, loss)
		# compute the norm
		norm_w2 = model.norm2()
		# compute the primal objective value
		primal = 0.5 * lambdaa * norm_w2 + c * loss_value
		return (primal, norm_w2, loss_value)

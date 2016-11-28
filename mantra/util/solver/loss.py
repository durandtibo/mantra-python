
from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):

	@abstractmethod
	def evaluate(self, model, x, y): 
		""" Evaluate the loss with the given model, pattern and label. The return value is a scalar.
		- model: model 
		- x: pattern
		- y: label """
		pass

	@abstractmethod
	def compute_gradient(self, model, x, y): 
		""" Compute the gradient of the loss with the given model, pattern and label. The return value is a vector.
		- model: model 
		- x: pattern
		- y: label """
		pass


	def __str__(self):
		return "Loss"



class HingeLoss(Loss):

	def __str__(self):
		return "HingeLoss"


	def evaluate(self, model, x, y): 
		""" Evaluate the hinge loss for the given model, pattern and label. The return value is a scalar.
		- model: model 
		- x: pattern
		- y: label """
		return max(0.0, 1 - model.value_of(x, y))


	def compute_gradient(self, model, x, y): 
		""" Compute the gradient of the hinge loss for the given model, pattern and label. The return value is a vector.
		- model: model 
		- x: pattern
		- y: label """
		score = model.value_of(x, y)
		if score < 1:
			return -y * x
		else:
			return np.zeros(x.size)
		



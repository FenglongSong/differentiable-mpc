from abc import ABC, abstractmethod


class DynamicalSystemBase(ABC):
	"""
	The base class for all dynamical systems.
	"""
	def __init__(self):
		self.name = None
		self.x = None
		self.u = None
		self.p = None

	@abstractmethod
	def dynamics(self, x, u, p): pass

	def nonlinear_reference(self, x, u, p):
		return None

	def external_cost(self, x, u, p):
		return None

	def nonlinear_path_constraints(self, x, u, p):
		return None

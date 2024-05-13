import numpy as np
import casadi as ca
import torch

from dynamical_system_assets.dynamical_system_base import DynamicalSystemBase


class Pendulum(DynamicalSystemBase):
	def __init__(self, mass: float = 1.0, length: float = 1.0, gravity: float = 9.81):
		super().__init__()

		# set up states and inputs
		theta = ca.SX.sym('theta')
		omega = ca.SX.sym('omega')
		self.x = ca.vertcat(theta, omega)

		tau = ca.SX.sym('tau')
		self.u = ca.vertcat(tau)

		# setup parameters
		self.l = length  # [m]
		self.m = mass  # [kg]
		self.g = gravity  # [m/s^2]

	def dynamics(self, x, u, p=None):
		x_dot = ca.vertcat(x[1], -self.g / self.l * ca.sin(x[0]) + u / self.m / self.l ** 2)
		return x_dot

	def nonlinear_reference(self, x, u, p):
		if isinstance(x, np.ndarray):
			return np.cos(x)
		elif isinstance(x, torch.Tensor):
			return torch.cos(x)
		elif isinstance(x, ca.casadi.SX) or isinstance(x, ca.casadi.MX):
			return ca.cos(x)


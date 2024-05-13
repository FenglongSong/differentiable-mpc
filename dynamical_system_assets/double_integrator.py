import pdb

import numpy as np
import casadi as ca

from dynamical_system_assets.dynamical_system_base import DynamicalSystemBase


class DoubleIntegrator(DynamicalSystemBase):
	def __init__(self, dim: int = 1, m=1.0, damping=0.0, f_max=100.0):
		super().__init__()
		self.dim = dim

		# set up states and inputs
		pos = ca.SX.sym('pos', dim)
		vel = ca.SX.sym('vel', dim)
		self.x = ca.vertcat(pos, vel)

		force = ca.SX.sym('force', dim)
		self.u = ca.vertcat(force)

		mass = ca.SX.sym('mass')
		damping = ca.SX.sym('damping')
		self.p = ca.vertcat(mass, damping)

		# setup parameters
		self.m = mass  # kg
		self.f_max = f_max  # N

	def dynamics(self, x, u, p):
		pos_dot = x[self.dim:self.dim*2]
		vel_dot = (u - p[1]*x[1:self.dim*2:2]) / p[0]  # only considered 1dim
		x_dot = ca.vertcat(pos_dot, vel_dot)
		return x_dot

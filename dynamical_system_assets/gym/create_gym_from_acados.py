import pdb

import gymnasium as gym
import numpy as np
import casadi as ca
from typing import Optional, Union
from stable_baselines3.common.env_checker import check_env
from acados_template import AcadosOcp, AcadosSim, AcadosSimSolver


class DynamicalSystemAcadosEnv(gym.Env):
	"""
	Create dynamical system environment for an acados model.
	"""
	INFINITY = 1e8  # use a large enough number as infinity, as the tradition in acados

	def __init__(
			self,
			acados_ocp: AcadosOcp,
			acados_sim: AcadosSim,
			integrator: str = 'rk4',
			noise_lb: float = -0.,
			noise_ub: float = +0.
	):

		self.acados_ocp = acados_ocp
		self.acados_sim = acados_sim
		self.acados_sim_solver = AcadosSimSolver(acados_sim)
		self.integrator = integrator
		self.noise_lb = noise_lb
		self.noise_ub = noise_ub

		nu = acados_ocp.model.u.shape[0]
		nx = acados_ocp.model.x.shape[0]

		action_space_lb = -self.INFINITY * np.ones(nu)
		action_space_ub = self.INFINITY * np.ones(nu)
		observation_space_lb = -self.INFINITY * np.ones(nx)
		observation_space_ub = self.INFINITY * np.ones(nx)
		if acados_ocp.constraints.idxbu.size:
			action_space_lb[acados_ocp.constraints.idxbu] = acados_ocp.constraints.lbu
			action_space_ub[acados_ocp.constraints.idxbu] = acados_ocp.constraints.ubu

		self.action_space = gym.spaces.Box(low=action_space_lb, high=action_space_ub, dtype=np.float32)
		self.observation_space = gym.spaces.Box(low=observation_space_lb, high=observation_space_ub, dtype=np.float32)
		self.state = (acados_ocp.constraints.lbx_0 + acados_ocp.constraints.ubx_0) / 2

		# TODO: deal with system dynamics to rely only on numpy operation, rather than using AcadosSim
		x = acados_ocp.model.x
		u = acados_ocp.model.u
		p = acados_ocp.model.p
		if acados_ocp.solver_options.integrator_type == 'DISCRETE':
			dynamics_discrete_time = casadi_expr_to_callable('f_disc', [x, u, p], [acados_ocp.model.dyn_disc_fun])
		elif acados_ocp.solver_options.integrator_type == 'ERK':
			dynamics_continuous_time = casadi_expr_to_callable('f_cont', [x, u, p], [acados_ocp.model.f_expl_expr])

		# stage cost
		if self.acados_ocp.cost.cost_type == 'NONLINEAR_LS' and self.acados_ocp.model.cost_y_expr is not None:
			self.cost_y_expr = casadi_expr_to_callable('cost_y_expr', [x, u, p], [self.acados_ocp.model.cost_y_expr])
		elif self.acados_ocp.cost.cost_type == 'EXTERNAL' and self.acados_ocp.model.cost_expr_ext_cost is not None:
			self.cost_expr_ext_cost = casadi_expr_to_callable('cost_expr_ext_cost', [x, u, p], [self.acados_ocp.model.cost_expr_ext_cost])

		self.dt = self.acados_ocp.solver_options.tf / self.acados_ocp.dims.N

	def step(self, action):
		assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
		assert self.state is not None, "Call reset before using step method."

		self.state = self.acados_sim_solver.simulate(x=self.state, u=action, p=None)
		noise = np.random.uniform(self.noise_lb, self.noise_ub, self.observation_space.shape)
		self.state += noise

		reward = self._reward_fn(self.state, action)

		return np.array(self.state, dtype=np.float32), reward, False, False, {}

	def reset(
			self,
			*,
			seed: Optional[int] = None,
			options: Optional[dict] = None,
	):
		super().reset(seed=seed)
		assert np.all(self.acados_ocp.constraints.lbx_0 == self.acados_ocp.constraints.ubx_0)

		self.state = np.array(self.acados_ocp.constraints.lbx_0, dtype=np.float32)
		return np.array(self.state, dtype=np.float32), {}

	def _reward_fn(self, state, action):
		stage_cost = None
		if self.acados_ocp.cost.cost_type == 'LINEAR_LS':
			error = self.acados_ocp.cost.Vx @ state + self.acados_ocp.cost.Vu @ action - self.acados_ocp.cost.yref
			stage_cost = self.dt * np.dot(error, self.acados_ocp.cost.W @ error)
		elif self.acados_ocp.cost.cost_type == 'NONLINEAR_LS':
			# TODO: test
			error = self.cost_y_expr([self.state, action, self.acados_ocp.parameter_values]) - self.acados_ocp.cost.yref
			stage_cost = self.dt * np.dot(error, self.acados_ocp.cost.W @ error)
		elif self.acados_ocp.cost.cost_type == 'EXTERNAL':
			stage_cost = self.dt * self.cost_expr_ext_cost([self.state, action, self.acados_ocp.parameter_values])

		slack_penalty_cost = self._path_constraint_violation_penalty(self.state)

		return -sum([stage_cost, slack_penalty_cost])

	def _path_constraint_violation_penalty(self, obs):
		idxsbx = self.acados_ocp.constraints.idxsbx
		idxsh = self.acados_ocp.constraints.idxsh  # TODO: implement the nonlinear constraints penalty
		idxsg = self.acados_ocp.constraints.idxsg  # TODO:
		slack_penalty = 0.
		if idxsbx.size:
			lbx_violation = np.clip(self.acados_ocp.constraints.lbx - obs, a_min=0., a_max=None)
			ubx_violation = np.clip(obs - self.acados_ocp.constraints.ubx, a_min=0., a_max=None)
			slack_penalty = 1/2 * np.dot(lbx_violation, self.acados_ocp.cost.Zl @ lbx_violation) + \
							1/2 * np.dot(ubx_violation, self.acados_ocp.cost.Zu @ ubx_violation) + \
							np.dot(self.acados_ocp.cost.zl, lbx_violation) + \
							np.dot(self.acados_ocp.cost.zu, ubx_violation)
		return slack_penalty


def casadi_expr_to_callable(name: str, inputs: list[ca.casadi.SX], outputs: list):
	"""
	Convert mapping from CasADi expression to a lambda expression which returns a numpy array
	:param name:
	:param inputs:
	:param outputs:
	:return:
	"""
	f = ca.Function(name, inputs, outputs)
	return lambda x: f(x).full().flatten()

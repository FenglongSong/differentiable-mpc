import pdb
import numpy as np
from typing import Any, Tuple, Union, Optional, Dict
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from gymnasium import spaces

from utils.mpc.mpc_layer import AcadosMpcLayer


class MpcLayerPolicy(BasePolicy):
	"""
	Define a neural network with a mpc layer.
	"""
	def __init__(self, input_dim: int, acados_ocp_layer: AcadosMpcLayer, hidden_dim: int = 32,
				 num_hidden_layers: int = 2, act=F.relu) -> None:
		super().__init__()
		self.input_dim = input_dim
		# Define the input layer
		self.input_layer = nn.Linear(input_dim, hidden_dim)

		# Define the hidden layers
		self.hidden_layers = nn.ModuleList()
		for _ in range(num_hidden_layers):
			self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

		# Define the layer that gives mpc parameters
		self.mpc_param_layer = nn.Linear(hidden_dim, acados_ocp_layer.np)

		# Define the differentiable mpc layer
		self.mpc_solver_layer = acados_ocp_layer

		# Define activation function
		self.activation = act

	def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
		"""
		Get the action according to the policy for a given observation.

		By default provides a dummy implementation -- not all BasePolicy classes
		implement this, e.g. if they are a Critic in an Actor-Critic method.

		:param observation:
		:param deterministic: Whether to use stochastic or deterministic actions
		:return: Taken action according to the policy
		"""
		x = observation
		# Forward pass through the network
		x = self.activation(self.input_layer(x))
		for hidden_layer in self.hidden_layers:
			x = self.activation(hidden_layer(x))
		mpc_params = self.mpc_param_layer(x)
		action = self.mpc_solver_layer(observation, mpc_params)
		return action


class ActorCriticMpcLayerPolicy(ActorCriticPolicy):
	"""
	Define a neural network with a differentiable MPC as the last layer.

	The network has 3 parts:
	- Fully-connected layers
	- Mpc-param layer: a linear layer which

	"""
	def __init__(
		self,
		observation_space: spaces.Space,
		action_space: spaces.Space,
		lr_schedule: Schedule,
		input_dim: int,
		acados_ocp_layer: AcadosMpcLayer,
		hidden_dim: int = 32,
		num_hidden_layers: int = 2,
		act=F.relu,
		**kwargs
	) -> None:

		super().__init__(
			observation_space=observation_space,
			action_space=action_space,
			lr_schedule=lr_schedule,
			**kwargs
		)

		self.input_dim = input_dim
		# Define the input layer
		self.input_layer = nn.Linear(input_dim, hidden_dim)

		# Define the hidden layers
		self.hidden_layers = nn.ModuleList()
		for _ in range(num_hidden_layers):
			self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

		# Define the layer that gives mpc parameters
		self.mpc_param_layer = nn.Linear(hidden_dim, acados_ocp_layer.np)

		# Define the differentiable mpc layer
		self.mpc_solver_layer = acados_ocp_layer

		# Define activation function
		self.activation = act

	def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
		"""
		Get the action according to the policy for a given observation.

		By default provides a dummy implementation -- not all BasePolicy classes
		implement this, e.g. if they are a Critic in an Actor-Critic method.

		:param observation:
		:param deterministic: Whether to use stochastic or deterministic actions
		:return: Taken action according to the policy
		"""
		x = observation.type(th.float32)
		x = self.activation(self.input_layer(x))
		for hidden_layer in self.hidden_layers:
			x = self.activation(hidden_layer(x))
		# mpc_params = F.relu(self.mpc_param_layer(x)) + 1e-2
		mpc_params = F.relu(self.mpc_param_layer(x)) + 1e-2
		action = self.mpc_solver_layer(observation, mpc_params)
		return action


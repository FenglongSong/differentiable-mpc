import pdb
from typing import Any, Union, List, Optional, Tuple, Dict, Type
import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from abc import ABC, abstractmethod
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from gymnasium import spaces

import acados_template


class BaseMpcSolver(ABC):
    """
    Base class for MPC Solvers.
    """

    def __init__(self):
        self.nx = None
        self.nu = None
        self.np = None
        pass

    @abstractmethod
    def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Solve the MPC problem for the given initial states.
        :param x0: given initial states
        :return: the optimal input at current time step and solver works successfully or not.
        """
        pass


class AcadosMpcSolver(BaseMpcSolver):
    """
    Acados as MPC Solvers.
    """
    def __init__(self, acados_ocp_solver: acados_template.AcadosOcpSolver):
        super(BaseMpcSolver, self).__init__()
        self.acados_ocp_solver = acados_ocp_solver
        self.nx = acados_ocp_solver.acados_ocp.dims.nx
        self.nu = acados_ocp_solver.acados_ocp.dims.nu
        self.np = acados_ocp_solver.acados_ocp.dims.np

    def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, bool]:
        u0 = self.acados_ocp_solver.solve_for_x0(x0)
        status = self.acados_ocp_solver.get_status()
        if status in [0, 2]:
            success = True
        else:
            success = False
        return u0, success


class MpcPolicy(BasePolicy, ABC):
    """Model predictive control as an expert.

    See `BaseAlgorithm` for more attributes.
    """
    mpc_solver: BaseMpcSolver

    def __init__(
            self,
            ocp_solver: BaseMpcSolver,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            features_extractor: Optional[BaseFeaturesExtractor] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            squash_output: bool = False
    ):
        super(MpcPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        self.mpc_solver = ocp_solver

    def _predict(
        self,
        observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this corresponds to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # We do not deal with recurrent states here, so only observation is useful now
        if not deterministic:
            RuntimeWarning("MPC policy can only return deterministic actions.")
        # pdb.set_trace()
        action, success = self.mpc_solver.solve(observation.cpu().numpy().reshape(-1,))
        if not success:
            raise RuntimeError("MPC Expert encounters a solver failure.")
        return th.tensor(action, dtype=th.float32)


class AcadosMpcPolicy(MpcPolicy):
    """Model predictive control as an expert.

    See `BaseAlgorithm` for more attributes.

    Args:
        env: EV charging environment
        lookahead: number of timesteps to forecast future trajectory

    Attributes:
        lookahead: number of timesteps to forecast future trajectory. Note that
            MPC cannot see future car arrivals and does not take them into
            account.
    """
    mpc_solver: AcadosMpcSolver

    def __init__(
            self,
            acados_ocp_solver: acados_template.AcadosOcpSolver,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            features_extractor: Optional[BaseFeaturesExtractor] = None,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            squash_output: bool = False
    ):
        super(MpcPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )
        self.mpc_solver = AcadosMpcSolver(acados_ocp_solver)


class ActorCriticMpcPolicy(ActorCriticPolicy):
    """
    MPC as a Actor-Critic policy.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[th.nn.Module] = th.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
from typing import Optional, Tuple, Dict, Type
import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from abc import ABC
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from gymnasium import spaces

from utils.mpc.mpc_solver import BaseMpcSolver


class MpcPolicy(BasePolicy, ABC):
    """Model predictive control as an expert.

    See `BaseAlgorithm` for more attributes.
    """
    mpc_solver: BaseMpcSolver

    def __init__(
            self,
            *args,
            mpc_solver: BaseMpcSolver,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mpc_solver = mpc_solver

    def _predict(
        self,
        observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # We do not deal with recurrent states here, so only observation is useful now
        if not deterministic:
            RuntimeWarning("MPC policy can only return deterministic actions.")
        action, _, success = self.mpc_solver.solve(observation.cpu().numpy().reshape(-1,))
        if not success:
            raise RuntimeError("MPC Expert encounters a solver failure.")
        return th.tensor(action, dtype=th.float32)


class ActorCriticMpcPolicy(MpcPolicy, ActorCriticPolicy):
    """
    MPC as an Actor-Critic policy.
    """
    def __init__(
        self,
        *args,
        mpc_solver: BaseMpcSolver,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        **kwargs
    ):
        super().__init__(
            *args,
            mpc_solver=mpc_solver,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            **kwargs
        )

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """

        raise NotImplementedError("evaluate_actions not implemented for MPCActorCriticPolicy")

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        action, opt_cost, success = self.mpc_solver.solve(obs.cpu().numpy().reshape(-1, ))
        if not success:
            raise RuntimeError("MPC Expert encounters a solver failure.")
        return -th.tensor(opt_cost, dtype=th.float32)

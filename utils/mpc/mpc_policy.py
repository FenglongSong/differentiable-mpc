from typing import Optional, Tuple, Dict, Type
import torch as th
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from abc import ABC
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from gymnasium import spaces

from utils.mpc.mpc_solver import BaseMpcSolver


class MpcPolicy(BasePolicy, ABC):
    """The MPC policy object.

    Parameters are mostly the same as `BasePolicy`; additions are documented below.

    :param args: positional arguments passed through to `BasePolicy`.
    :param kwargs: keyword arguments passed through to `BasePolicy`.
    :param mpc_solver: A mpc solver which solves an optimal control problem to get action.
    """
    mpc_solver: BaseMpcSolver

    def __init__(
            self,
            mpc_solver: BaseMpcSolver,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mpc_solver = mpc_solver

    def _predict(
        self,
        observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        The action is computed by calling the MPC solver to solve the underlying MPC problem.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        # We do not deal with recurrent states here, so only observation is useful now
        if not deterministic:
            RuntimeWarning("MPC policy can only return deterministic actions.")
        state = self._state_estimation(observation)
        action, _, success = self.mpc_solver.solve(state.cpu().numpy().reshape(-1,))
        if not success:
            raise RuntimeError("MPC Expert encounters a solver failure.")
        return th.tensor(action, dtype=th.float32)
    
    def _state_estimation(self, observation: PyTorchObs):
        """
        Estimate states from observation. Can be cusomized.
        """
        return observation



class ActorCriticMpcPolicy(ActorCriticPolicy):
    """
    MPC as an Actor-Critic policy.
    """
    def __init__(
        self,
        mpc_solver: BaseMpcSolver,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *args,
        **kwargs
    ):
        super().__init__(
            mpc_solver=mpc_solver,
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs
        )

    def _predict(
        self,
        observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        The action is computed by calling the MPC solver to solve the underlying MPC problem.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        # We do not deal with recurrent states here, so only observation is useful now
        if not deterministic:
            RuntimeWarning("MPC policy can only return deterministic actions.")
        action, _, success = self.mpc_solver.solve(observation.cpu().numpy().reshape(-1,))
        if not success:
            raise RuntimeError("MPC Expert encounters a solver failure.")
        return th.tensor(action, dtype=th.float32)

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

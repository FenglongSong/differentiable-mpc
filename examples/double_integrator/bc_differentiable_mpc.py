"""This is a simple example demonstrating how to clone the behavior of an expert.

Original code comes from here: https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py
https://imitation.readthedocs.io/en/latest/tutorials/10_train_custom_env.html

REFER TO THIS ONE: https://github.com/MPC-Based-Reinforcement-Learning/rlmpc/blob/main/rlmpc/gym/continuous_cartpole/environment.py

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import pdb
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("./")
from gymnasium.wrappers import TimeLimit
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
from rl_zoo3.utils import linear_schedule
import gymnasium as gym
import matplotlib.pyplot as plt

from utils.mpc.mpc_solver import AcadosMpcSolver
from utils.mpc.mpc_policy import MpcPolicy
from utils.mpc.mpc_layer import AcadosMpcLayer
from utils.mpc.mpc_network import ActorCriticMpcLayerPolicy
from dynamical_system_assets.double_integrator.acados_modules import create_acados_ocp_solver
from dynamical_system_assets.double_integrator.gym_environment import DoubleIntegratorEnv


def get_trajectory(n_steps: int, env: gym.Env, policy: BasePolicy):
    obs, _ = env.reset()
    obs_traj = np.zeros((env.observation_space.shape[0], n_steps))
    act_traj = np.zeros((env.action_space.shape[0], n_steps))
    for i_step in range(n_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs_traj[:, i_step] = obs
        act_traj[:, i_step] = action
        next_obs, reward, done, _, info = env.step(action)
        obs = next_obs
    return obs_traj, act_traj


def main():
    # Create a single environment for training with SB3
    env = DoubleIntegratorEnv()
    env = TimeLimit(env, max_episode_steps=500)

    def _make_env():
        """
        Helper function to create a single environment. Put any logic here, but make sure to return a RolloutInfoWrapper.
        """
        _env = DoubleIntegratorEnv()
        _env = TimeLimit(_env, max_episode_steps=500)
        _env = RolloutInfoWrapper(_env)
        return _env
    venv = DummyVecEnv([_make_env for _ in range(1)])  # ! can only set 1 venv since the MpcSolver class is not designed for batch operations yet
    rng = np.random.default_rng()

    expert_acados_ocp_solver = create_acados_ocp_solver()
    mpc_expert = MpcPolicy(AcadosMpcSolver(expert_acados_ocp_solver), env.observation_space, env.action_space)

    student_acados_ocp_solver = create_acados_ocp_solver()
    student_policy = ActorCriticMpcLayerPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=linear_schedule(1e-3),
        input_dim=2,
        acados_ocp_layer=AcadosMpcLayer(student_acados_ocp_solver),
        hidden_dim=64,
        num_hidden_layers=3,
    )

    reward, _ = evaluate_policy(mpc_expert, env, 10, deterministic=True)
    print(f"Expert reward: {reward}")

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        mpc_expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=student_policy,
        l2_weight=1e-4
    )

    print("Evaluating the untrained policy.")
    reward, _ = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=10,
        render=False,
    )
    print(f"Reward before training: {reward}")
    
    print(f"MPC parameter before training: {student_policy.get_mpc_params(torch.zeros(1, 2))}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=10, progress_bar=False)

    print("Evaluating the trained policy.")
    reward, _ = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=10,
        render=False,
        deterministic=True
    )
    print(f"Reward after training: {reward}")
    print(f"MPC parameter after training: {student_policy.get_mpc_params(torch.zeros(1, 2))}")


    # visualization
    obs_traj, action_traj = get_trajectory(300, env, mpc_expert)
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(obs_traj[0, :], label='pos')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(obs_traj[1, :], label='vel')
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(action_traj[0, :], label='force')
    ax[2].grid()
    ax[2].legend()
    fig.suptitle('Closed-loop performance of expert policy')

    bc_trainer.policy.eval()
    obs_traj, action_traj = get_trajectory(300, env, bc_trainer.policy)
    fig, ax = plt.subplots(3, sharex=True)
    ax[0].plot(obs_traj[0, :], label='pos')
    ax[0].grid()
    ax[0].legend()
    ax[1].plot(obs_traj[1, :], label='vel')
    ax[1].grid()
    ax[1].legend()
    ax[2].plot(action_traj[0, :], label='force')
    ax[2].grid()
    ax[2].legend()
    fig.suptitle('Closed-loop performance of learned policy')
    plt.show()


if __name__ == "__main__":
    main()

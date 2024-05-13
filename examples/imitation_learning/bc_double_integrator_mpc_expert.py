"""This is a simple example demonstrating how to clone the behavior of an expert.

Original code comes from here: https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py
https://imitation.readthedocs.io/en/latest/tutorials/10_train_custom_env.html

REFER TO THIS ONE: https://github.com/MPC-Based-Reinforcement-Learning/rlmpc/blob/main/rlmpc/gym/continuous_cartpole/environment.py

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import pdb
import sys
sys.path.append("./")
from gymnasium.wrappers import TimeLimit
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.algorithms import bc
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
import gymnasium as gym
import matplotlib.pyplot as plt

from dynamical_system_assets.gym.create_gym_from_acados import DynamicalSystemAcadosEnv
from utils.mpc.mpc_solver import AcadosMpcSolver
from utils.mpc.mpc_policy import MpcPolicy
from dynamical_system_assets.acados_modules.create_acados_modules import *


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
    double_integrator = DoubleIntegrator()
    config_file = './dynamical_system_assets/config/double_integrator_config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    acados_ocp = create_acados_ocp(double_integrator, config)
    acados_sim = create_acados_sim(double_integrator, config)

    # Create a single environment for training with SB3
    env = DynamicalSystemAcadosEnv(acados_ocp, acados_sim)
    env = TimeLimit(env, max_episode_steps=500)

    # Create a vectorized environment for training with `imitation`

    # Option A: use a helper function to create multiple environments
    def _make_env():
        """
        Helper function to create a single environment. Put any logic here, but make sure to return a RolloutInfoWrapper.
        """
        _env = DynamicalSystemAcadosEnv(acados_ocp, acados_sim)
        _env = TimeLimit(_env, max_episode_steps=500)
        _env = RolloutInfoWrapper(_env)
        return _env
    venv = DummyVecEnv([_make_env for _ in range(1)])
    rng = np.random.default_rng()

    acados_ocp_solver = create_acados_ocp_solver(double_integrator, config)
    mpc_expert = MpcPolicy(AcadosMpcSolver(acados_ocp_solver), env.observation_space, env.action_space)
    reward, _ = evaluate_policy(mpc_expert, env, 10)
    print(f"Expert reward: {reward}")

    def sample_expert_transitions():
        print("Sampling expert transitions.")
        rollouts = rollout.rollout(
            mpc_expert,
            venv,
            rollout.make_sample_until(min_timesteps=None, min_episodes=50),
            rng=rng,
        )
        return rollout.flatten_trajectories(rollouts)

    transitions = sample_expert_transitions()
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )

    print("Evaluating the untrained policy.")
    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=3,
        render=False,  # comment out to speed up
    )
    print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=5)

    print("Evaluating the trained policy.")
    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=10,
        render=False,  # comment out to speed up
    )
    print(f"Reward after training: {reward}")

    obs_traj, action_traj = get_trajectory(100, env, bc_trainer.policy)
    # obs_traj, action_traj = get_trajectory(100, env, mpc_expert)
    plt.figure(1)
    plt.plot(obs_traj[0, :])
    plt.plot(obs_traj[1, :])
    plt.grid()

    plt.figure(2)
    plt.plot(action_traj[0, :])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

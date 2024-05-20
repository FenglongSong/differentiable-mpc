# # Double Integrator
# 
# Descriptions about Pendulum environment: https://gymnasium.farama.org/environments/classic_control/pendulum/
# 
# The Hyperparameters are taken from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
# 

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.ppo.policies import MlpPolicy
from gymnasium.wrappers import TimeLimit

import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

from dynamical_system_assets.double_integrator.gym_environment import DoubleIntegratorEnv

# Create the Gym env and instantiate the agent
env = DoubleIntegratorEnv()
env = TimeLimit(env, max_episode_steps=100)
model = PPO(
    MlpPolicy,
    env, 
    verbose=0, 
    n_steps=1024,
    gae_lambda=0.95,
    gamma=0.95,
    n_epochs=10,
    ent_coef=0.0,
    learning_rate=2e-3,
    clip_range=0.2,
    use_sde=True,
    sde_sample_freq=4)


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"Before training: mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Train the agent
print("Training...")
model.learn(total_timesteps=100_000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"After training: mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")



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

# visualization
import matplotlib.pyplot as plt
obs_traj, action_traj = get_trajectory(500, env, model.policy)
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
fig.suptitle('Closed-loop performance of trained policy')
plt.show()

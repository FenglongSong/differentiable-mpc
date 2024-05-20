"""This is a simple example demonstrating how to clone the behavior of an expert.

Original code comes from here: https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py
https://imitation.readthedocs.io/en/latest/tutorials/10_train_custom_env.html

REFER TO THIS ONE: https://github.com/MPC-Based-Reinforcement-Learning/rlmpc/blob/main/rlmpc/gym/continuous_cartpole/environment.py

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.algorithms import bc


from dynamical_system_assets.double_integrator.gym_environment import DoubleIntegratorEnv

def main():
    env_name = "custom/DoubleIntegrator-v0"
    gym.register(
        id=env_name,
        entry_point=DoubleIntegratorEnv,
        max_episode_steps=500,
    )
    rng = np.random.default_rng(0)
    env = make_vec_env(
        env_name,
        rng=rng,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    )

    ppo_expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=50,
        n_steps=64,
    )
    reward, _ = evaluate_policy(ppo_expert, env, 10)
    print(f"Reward before training: {reward}")
    ppo_expert.learn(10000)  # Note: set to 100000 to train a proficient expert
    reward, _ = evaluate_policy(ppo_expert, ppo_expert.get_env(), 10)
    print(f"Expert reward: {reward}")

    def sample_expert_transitions():
        print("Sampling expert transitions.")
        rollouts = rollout.rollout(
            ppo_expert,
            env,
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
    bc_trainer.train(n_epochs=1)

    print("Evaluating the trained policy.")
    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=3,
        render=False,  # comment out to speed up
    )
    print(f"Reward after training: {reward}")


if __name__ == "__main__":
    main()

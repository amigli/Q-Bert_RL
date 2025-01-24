import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from EnvironmentWrappers.ObsRewardWrapper import ObsRewardWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv

gym.register_envs(ale_py)

def make_env(env_id):
    def _init():
        env = gym.make(env_id)  
        env = ObsRewardWrapper(env)
        return env
    return _init


if __name__ == '__main__':
    num_envs = 1
    envs = SubprocVecEnv([make_env("ALE/Qbert-ram-v5") for _ in range(num_envs)])

    model = A2C("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=100)

    print("Training terminato")

    # Valutare il modello
    mean_reward, std_reward = evaluate_policy(model, envs, n_eval_episodes=10)
    print(f"Ricompensa media: {mean_reward:.2f}, deviazione standard: {std_reward:.2f}")
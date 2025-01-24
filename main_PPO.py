import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import ale_py
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from EnvironmentWrappers.ObsRewardWrapper import ObsRewardWrapper

gym.register_envs(ale_py)

## TRAINING 
env = gym.make("ALE/Qbert-ram-v5")  
env = ObsRewardWrapperc(env)

model = PPO(
    "MlpPolicy",      # Tipo di rete neurale (MLP per osservazioni vettoriali)
    env,              # Ambiente
    verbose=1,        # Livello di logging
    learning_rate=0.0003,  # Tasso di apprendimento
    n_steps=2048,     # Passi raccolti per ogni aggiornamento
    batch_size=64,    # Dimensione del batch per l'ottimizzazione
    n_epochs=1000,      # Numero di epoche per aggiornare il modello
    gamma=0.99,       # Fattore di sconto
)

model.learn(total_timesteps = 10000)

print("Training terminato")

# Valutare il modello
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Ricompensa media: {mean_reward:.2f}, deviazione standard: {std_reward:.2f}")
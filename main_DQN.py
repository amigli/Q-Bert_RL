import gymnasium as gym
import ale_py
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from EnvironmentWrappers.ObsRewardWrapper import ObsRewardWrapper
from Algorithms.QBertAgent_DQN import DQNAgent
from EnvironmentWrappers.utils import salva_csv


gym.register_envs(ale_py)

learning_rate = 0.01
n_episodes = 10
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  
final_epsilon = 0.1

env = gym.make("ALE/Qbert-ram-v5")  
env =  ObsRewardWrapper(env)
obs, info = env.reset()

reward_per_episode = []
step_per_episode = []
epsilon_value = []


state_dim = obs.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)

# TRAINING
batch_size = 32
num_episodes = 1000
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    total_reward = 0
    total_step = 0
    while not done:
        action = agent.act(obs, True)
        next_obs, reward, terminated, truncated, info = env.step(action)
    
        done =  terminated or truncated    
        
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        total_step +=1
        agent.replay(batch_size)
    reward_per_episode.append(total_reward)
    step_per_episode.append(total_step)
    epsilon_value.append(agent.epsilon)

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

env.close()

salva_csv(reward_per_episode, "Reward", "csv_reward_DQN.csv")
salva_csv(step_per_episode, "Steps", "csv_steps_DQN.csv")
salva_csv(epsilon_value, "Epsilon", "csv_epsilon_DQN.csv")



num_eval_episodes = 10

env = gym.make("ALE/Qbert-ram-v5", render_mode="rgb_array")  
env =  ObsRewardWrapper(env)  
env = RecordVideo(env, video_folder="videos_first", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)


for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        obs =  next_obs

        done = terminated or truncated
env.close()
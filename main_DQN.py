import gymnasium as gym
import ale_py
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from classes.QbertObservationWrapper import QbertObservationWrapperBox
from classes.QBertAgent_DQN import DQNAgent
from classes.RewardFunction import RewardFunction
gym.register_envs(ale_py)
# hyperparameters

learning_rate = 0.01
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  
final_epsilon = 0.1

env = gym.make("ALE/Qbert-ram-v5")  
env =  QbertObservationWrapperBox(env)
obs, info = env.reset()

state_dim = obs[2].shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=10000)

# Train the DQN agent with Experience Replay Buffer
batch_size = 32
num_episodes = 1000
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    rewardFunction = RewardFunction(obs[0])
    obs = obs[2]
    total_reward = 0
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
    
        reward = rewardFunction.calculate_reward(next_obs, reward)
        next_obs = next_obs[2]

        done =  terminated or truncated    
        
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        agent.replay(batch_size)
        
    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

"""
## TRAINING 
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    rewardFunction = RewardFunction(obs[0])
    obs = obs[1]
    while not done:
        action = 3

        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = rewardFunction.calculate_reward(next_obs, reward)
        next_obs = next_obs[1]

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

env.close()


num_eval_episodes = 10

env = gym.make("ALE/Qbert-ram-v5", render_mode="rgb_array")  
env =  QbertObservationWrapper(env)  
env = RecordVideo(env, video_folder="videos_first", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)


for episode_num in range(num_eval_episodes):
    obs, info = env.reset()
    rewardFunction = RewardFunction(obs[0])
    obs = obs[1]
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        reward = rewardFunction.calculate_reward(next_obs, reward)
        obs =  next_obs[1]

        done = terminated or truncated
env.close()"""
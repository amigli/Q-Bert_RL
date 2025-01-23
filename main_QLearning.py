from classes.QBertAgent import QBAgent
import gymnasium as gym
import ale_py
from tqdm import tqdm
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from classes.QbertObservationWrapper import QbertObservationWrapper
from classes.RewardFunction import RewardFunction
gym.register_envs(ale_py)
# hyperparameters
learning_rate = 0.01
n_episodes = 1000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  
final_epsilon = 0.1

## TRAINING 
env = gym.make("ALE/Qbert-ram-v5")  
env =  QbertObservationWrapper(env)
agent = QBAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

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
env.close()

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')

print("Training terminato")




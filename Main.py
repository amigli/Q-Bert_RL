from classes.QBertAgent import QBAgent
import gymnasium as gym
import ale_py
from tqdm import tqdm

gym.register_envs(ale_py)
# hyperparameters
learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make("ALE/Qbert-ram-v5")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = QBAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

history = []

for episode in range(n_episodes):
    obs, info = env.reset()
    done = False
    history.append(obs)
    print(obs) 

    obs = tuple(obs)
    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        history.append(next_obs)

        print(next_obs)
        # print(str(obs) + "- Lenght: " + str(len(obs)))
        # print(str(next_obs) +  "- Lenght: " + str(len(obs)))

        obs =  tuple(obs)
        next_obs = tuple(next_obs)
        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

"""
history_sets = []
history_count = []
for i in range(128):
    history_count.append(0)

for i in range(128):
    history_sets.append(set())

for j in range(128):
    for i in range(len(history)):
        elem = history[i][j]
        # print(j)
        history_sets[j].add(elem)

for i in range(128):
    print("Position:" + str(i) + ", count:" + str(len(history_sets[i])))

# print(history_sets)
"""

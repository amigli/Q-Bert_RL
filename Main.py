from classes.QBertAgent import QBAgent
import gymnasium as gym
import ale_py
from OC_Atari_Files.qbert import _detect_objects_ram, _init_objects_ram
from tqdm import tqdm
import pickle 
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from tqdm import tqdm


gym.register_envs(ale_py)
# hyperparameters
learning_rate = 0.01
n_episodes = 500
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

# history = []

for episode in tqdm(range(n_episodes)):
    obs_wrapper = _init_objects_ram()
    obs, info = env.reset()
    obs_wrapper = tuple( _detect_objects_ram(obs_wrapper, obs))
    done = False
    # history.append(obs)
    # print(obs) 

    
    # play one episode
    while not done:
        action = agent.get_action(obs_wrapper)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # history.append(next_obs)

        # print(next_obs)
        # print(str(obs) + "- Lenght: " + str(len(obs)))
        # print(str(next_obs) +  "- Lenght: " + str(len(obs)))

        obs_wrapper = _init_objects_ram()
        obs_wrapper =  tuple(_detect_objects_ram(obs_wrapper, obs))
        next_obs_wrapper = _init_objects_ram()
        next_obs_wrapper = tuple(_detect_objects_ram(next_obs_wrapper, next_obs))
        # update the agent
        agent.update(obs_wrapper, action, reward, terminated, next_obs_wrapper)


        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

num_eval_episodes = 4

env = gym.make("ALE/Qbert-ram-v5", render_mode="rgb_array")  # replace with your environment
env = RecordVideo(env, video_folder="videos_first", name_prefix="eval",
                  episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)

for episode_num in range(num_eval_episodes):
    obs_wrapper = _init_objects_ram()
    obs, info = env.reset()
    obs_wrapper = tuple( _detect_objects_ram(obs_wrapper, obs))
    done = False

    while not done:
        action = agent.get_action(obs_wrapper)
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        obs_wrapper = _init_objects_ram()
        obs_wrapper = tuple( _detect_objects_ram(obs_wrapper, obs))

        done = terminated or truncated
env.close()

print(f'Episode time taken: {env.time_queue}')
print(f'Episode total rewards: {env.return_queue}')
print(f'Episode lengths: {env.length_queue}')

"""
print("Training terminato")
file = open("saved_agent", "wb")
pickle.dump(agent.get_Qvalues(), file)
file.close()
print("Qu")
"""



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

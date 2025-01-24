import gymnasium as gym

class RewardFunctionWrapper(gym.Wrapper):
    def __init__(self, env, reward_function):
        super(RewardFunctionWrapper, self).__init__(env)
        self.reward_function = reward_function

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        reward = self.reward_function.calculate_reward(observation, reward)

        return observation, reward, done, info
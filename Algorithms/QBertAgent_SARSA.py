from collections import defaultdict
import gymnasium as gym
import numpy as np
import random as rand


class SARSAAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        
        self.lr = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        

    def get_action(self, obs, train) -> int:
        if train and np.random.random() < self.epsilon:
            if obs[0] != None and obs[0].xy == (74,17):
                sequence = [0, 3, 5]
                return rand.choice(sequence)
            else:
                sequence = [0, 2, 3, 4, 5]
                return rand.choice(sequence)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
        next_action: int,
    ):

        """Calcolo il TD error"""
        if not terminated:
            td_target = reward + self.discount_factor * self.q_values[next_obs][next_action]
            td_error = td_target - self.q_values[obs][action]
            self.q_values[obs][action] += self.lr * td_error


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def get_Qvalues(self):
        dictionary =  dict(self.q_values)
        return dictionary
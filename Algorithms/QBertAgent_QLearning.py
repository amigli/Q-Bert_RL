from collections import defaultdict
import gymnasium as gym
import numpy as np
import random as rand
import pickle

class QBAgent:
    def __init__(
        self,
        env: gym.Env,
        step_size: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        q_values = None
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.step_size = step_size
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool], train:bool) -> int:
        
        if train and  np.random.random() < self.epsilon:
            if obs[0] == 74 and obs[1] == 17:
                sequence = [3,5]
                return rand.choice(sequence)
            else:
                sequence = [0, 2, 3, 4, 5]
                return rand.choice(sequence)
        else:
            action = int(np.argmax(self.q_values[obs]))
            """ if not(train): print("Ho scelto il meglio!" + str(action))"""
            return action
    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):

        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.step_size * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_Qvalues(self):
        with open("q_values.pkl", "wb") as f:
            pickle.dump(dict(self.q_values), f)
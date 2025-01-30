from collections import defaultdict
import gymnasium as gym
import numpy as np
import pickle
import random as rand

class SARSALambdaAgent:
    def __init__(
        self,
        env: gym.Env,
        step_size: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        lambda_: float = 0.9,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.full(self.env.action_space.n, -100.0))

        self.step_size = step_size
        self.discount_factor = discount_factor
        self.lambda_ = lambda_

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.eligibility_traces = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs, train):
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
        obs,
        action,
        reward,
        terminated,
        next_obs,
        next_action
    ):

        td_error = (
            reward
            + self.discount_factor * self.q_values[next_obs][next_action] * (1 - terminated)
            - self.q_values[obs][action]
        )

        self.eligibility_traces[obs][action] += 1
        # Backwardview = aggiorno gli stati passati in base all'egibility trace
        for state_action, trace_value in self.eligibility_traces.items():
            self.q_values[state_action] += self.step_size * td_error * trace_value
            self.eligibility_traces[state_action] *= self.discount_factor * self.lambda_

        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_q_values(self, filename="q_values.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(dict(self.q_values), f)

    def load_q_values(self, filename="q_values.pkl"):
        with open(filename, "rb") as f:
            self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n), pickle.load(f))
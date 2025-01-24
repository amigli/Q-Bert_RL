class RewardFunction():
    def __init__(self, init_lives):
        self.lives = init_lives
    
    def calculate_reward(self, observation, reward):
        next_lives, obs =  observation[0], observation[1]

        if next_lives < self.lives :
            reward -= 20
            self.lives = next_lives
        
        if obs[0] != None:
            if obs[24] != None and obs[24].xy == obs[0].xy:
                reward -= 15
            if obs[25] != None and obs[25].xy == obs[0].xy:
                reward -= 15

        return reward
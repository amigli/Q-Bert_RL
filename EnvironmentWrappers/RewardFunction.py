class RewardFunction():
    def __init__(self, init_lives):
        self.lives = init_lives
    
    def calculate_reward(self, observation, reward):
        calculated_reward =  0

        match reward:
            case 25:
                calculated_reward = 1
            case 100:
                calculated_reward = 2
            case 300:
                calculated_reward = 2.5
            case 500:
                calculated_reward = 3
            case 3100:
                calculated_reward = 5
            case _:
                if reward > 0:
                    calculated_reward = 3.5 # Bonus che QBert può ricevere in alcuni round
                else:
                    calculated_reward = 0

        next_lives, obs =  observation[0], observation[1]

        if next_lives < self.lives :
            calculated_reward -= 3
            self.lives = next_lives
        if obs[0] != None:
            if obs[24] != None and obs[24].xy == obs[0].xy:
                calculated_reward -= 1
            if obs[25] != None and obs[25].xy == obs[0].xy:
                calculated_reward -= 1
                
        if calculated_reward == 0:
            # QBert non ha fatto nulla, si dà una piccola penalità per evitare che stia fermo ad aspettare (esegue NO_OP)
            calculated_reward -= -0.5

        return calculated_reward
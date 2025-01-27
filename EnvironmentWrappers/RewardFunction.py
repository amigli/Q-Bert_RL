class RewardFunction():
    def __init__(self, init_lives):
        self.lives = init_lives
    
    def calculate_reward(self, observation, reward):
        calculated_reward =  0
        # print("Reward ambiente:" + str(reward))
        match reward:
            case 25:
                calculated_reward = 1
            case 100:
                calculated_reward = 1.5
            case 300:
                calculated_reward = 2
            case 500:
                calculated_reward = 2.5
            case 3100:
                calculated_reward = 4
            case _:
                if reward > 0:
                    calculated_reward = 3.5 # Bonus che QBert può ricevere in alcuni round
                else:
                    calculated_reward = 0

        next_lives, obs =  observation[0], observation[1]
        match next_lives:
          case 2:
            next_lives = 4
          case 1:
            next_lives = 3
          case 0:
            next_lives =  2
          case 255:
            next_lives = 1
          case 254:
            next_lives = 0
  
        # print("next_lives:" + str(next_lives) + ", old_lives:" + str(self.lives))
        if next_lives < self.lives :
            calculated_reward = 0 - 5
            self.lives = next_lives
            # print("Morto")

        if next_lives == 0:
            calculated_reward = 0 - 10

        if obs[0] != None:
            if obs[24] != None and obs[24].xy == obs[0].xy:
                # print("Scontro con avversario 1")
                calculated_reward = 0 - 2
            if obs[25] != None and obs[25].xy == obs[0].xy:
                # print("Scontro con avversario 1")
                calculated_reward = 0 - 2
                
        """if calculated_reward == 0:
            # QBert non ha fatto nulla, si dà una piccola penalità per evitare che stia fermo ad aspettare (esegue NO_OP)
            calculated_reward = 0- 0.5"""
        # print(calculated_reward)
        return calculated_reward
import gymnasium as gym
from .RewardFunction import RewardFunction
from OC_Atari_Files.qbert import _detect_objects_ram, _init_objects_ram
import numpy as np


class ObsRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ObsRewardWrapper, self).__init__(env)
        self.reward_function = RewardFunction(init_lives=4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(75,), dtype=np.float32
        )

    def reset(self, **kwargs):
        # Richiama il reset originale
        observation, info = self.env.reset(**kwargs)
        obs = _init_objects_ram()
        obs = _detect_objects_ram(obs, observation, hud=False)
        newobs = []

        if obs[0] != None:
            # Inserimento delle coordinate del giocatore
            newobs.append(obs[0].xy[0])
            newobs.append(obs[0].xy[1])

            # Inserimento dei colori dei cubi:
            for i in range(21):
                j = i+1
                if obs[j] != None:
                    newobs.append(obs[j].rgb[0])
                    newobs.append(obs[j].rgb[1])
                    newobs.append(obs[j].rgb[2])
                else:
                    newobs.append(0)
                    newobs.append(0)
                    newobs.append(0)

            # Inserimento delle posizioni dei dischi di salvataggio
            if obs[23] != None :
                # Inserimento delle posizioni dei dischi di salvataggio
                newobs.append(obs[23].xy[0])
                newobs.append(obs[23].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[24] != None :
                # Inserimento delle coordinate di coily
                newobs.append(obs[24].xy[0])
                newobs.append(obs[24].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[25] != None :
                # Inserimento delle coordinate di Purple Ball
                newobs.append(obs[25].xy[0])
                newobs.append(obs[25].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[26] != None :
                # Inserimento delle coordinate di Purple Ball
                newobs.append(obs[26].xy[0])
                newobs.append(obs[26].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[26] != None :
                # Inserimento delle coordinate di Sam
                newobs.append(obs[27].xy[0])
                newobs.append(obs[27].xy[1])
            else: 
                newobs.append(0)
                newobs.append(0)
              
            newobs = np.array(newobs)
        else:
            newobs =  np.zeros(75)

        return newobs, info

        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs = _init_objects_ram()
        obs = _detect_objects_ram(obs, observation, hud=False)
        newobs = []

        if obs[0] != None:
            # Inserimento delle coordinate del giocatore
            newobs.append(obs[0].xy[0])
            newobs.append(obs[0].xy[1])

            # Inserimento dei colori dei cubi:
            for i in range(21):
                j = i+1
                if obs[j] != None:
                    newobs.append(obs[j].rgb[0])
                    newobs.append(obs[j].rgb[1])
                    newobs.append(obs[j].rgb[2])
                else:
                    newobs.append(0)
                    newobs.append(0)
                    newobs.append(0)

            # Inserimento delle posizioni dei dischi di salvataggio
            if obs[23] != None :
                # Inserimento delle posizioni dei dischi di salvataggio
                newobs.append(obs[23].xy[0])
                newobs.append(obs[23].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[24] != None :
                # Inserimento delle coordinate di coily
                newobs.append(obs[24].xy[0])
                newobs.append(obs[24].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[25] != None :
                # Inserimento delle coordinate di Purple Ball
                newobs.append(obs[25].xy[0])
                newobs.append(obs[25].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[26] != None :
                # Inserimento delle coordinate di Purple Ball
                newobs.append(obs[26].xy[0])
                newobs.append(obs[26].xy[1])
            else:
                newobs.append(0)
                newobs.append(0)

            if obs[26] != None :
                # Inserimento delle coordinate di Sam
                newobs.append(obs[27].xy[0])
                newobs.append(obs[27].xy[1])
            else: 
                newobs.append(0)
                newobs.append(0)
              
            newobs = np.array(newobs)
        else:
            newobs =  np.zeros(75)
        obs = (observation[8], obs, newobs)
        reward = self.reward_function.calculate_reward(obs, reward)

        return newobs, reward, terminated, truncated, info
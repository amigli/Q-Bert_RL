from gymnasium import ObservationWrapper
from OC_Atari_Files.qbert import _detect_objects_ram, _init_objects_ram
import numpy as np

class QbertObservationWrapperTuple(ObservationWrapper):
    def observation(self, observation):
        obs = _init_objects_ram()
        obs = tuple( _detect_objects_ram(obs, observation, hud=False))
        obs = (observation[8], obs)
        return obs

class QbertObservationWrapperBox(ObservationWrapper):
    def observation(self, observation):
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
                newobs.append(obs[j].rgb[0])
                newobs.append(obs[j].rgb[1])
                newobs.append(obs[j].rgb[2])

            # Inserimento delle posizioni dei dischi di salvataggio
            newobs.append(obs[23].xy[0])
            newobs.append(obs[23].xy[1])

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
        return obs
       
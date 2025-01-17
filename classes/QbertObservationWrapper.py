from gymnasium import ObservationWrapper
from OC_Atari_Files.qbert import _detect_objects_ram, _init_objects_ram

class QbertObservationWrapper(ObservationWrapper):
    def observation(self, observation):
        obs = _init_objects_ram()
        obs = tuple( _detect_objects_ram(obs, observation))

        return obs

        
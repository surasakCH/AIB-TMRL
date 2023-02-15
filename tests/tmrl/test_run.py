from tmrl import get_environment
from time import sleep
import numpy as np


# default LIDAR observations are of shape: ((1,), (4, 19), (3,), (3,))
# representing: (speed, 4 last LIDARs, 2 previous actions)
# actions are [gas, break, steer], analog between -1.0 and +1.0
def model(obs):
    """
    simplistic policy for LIDAR observations
    """
    deviation = obs[1].mean(0)
    deviation /= (deviation.sum() + 0.001)
    steer = 0
    for i in range(19):
        steer += (i - 9) * deviation[i]
    steer = - np.tanh(steer * 4)
    steer = min(max(steer, -1.0), 1.0)
    return np.array([1.0, 0.0, steer])

# Let us retrieve the TMRL Gym environment.
# The environment you get from get_environment() depends on the content of config.json
env = get_environment()

sleep(1.0)  # just so we have time to focus the TM20 window after starting the script

obs, info = env.reset()  # reset environment

import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='matlab-v0',
    entry_point='gym_matlab.envs:matlabEnv',
)

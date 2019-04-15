#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
matlab simulation environment.

Each episode is running an matlab simulation.
"""

# core modules
import logging.config
import math
import pkg_resources
import random

# 3rd party modules
from gym import spaces
import cfg_load
import gym
import numpy as np


path = 'config.yaml'  # always use slash in packages
filepath = pkg_resources.resource_filename('gym_matlab', path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config['LOGGING'])



class matlabEnv(gym.Env):
    """
    Define a simple matlab environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("matlabEnv - Version {}".format(self.__version__))

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        pass
        
    def _take_action(self, action):
        pass

    def _get_reward(self):
        """Reward is given for a sold matlab."""
        pass

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        pass

    def _render(self, mode='human', close=False):
        pass

    def _get_state(self):
        pass

    def seed(self, seed):
        pass

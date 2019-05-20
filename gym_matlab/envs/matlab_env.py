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


def matlab_function(x, y):
    #cost/error
    return np.power(x,2) + np.power(y, 2)

class matlabEnv(gym.Env):
    """
    Define a simple matlab environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("matlabEnv - Version {}".format(self.__version__))
        # General variables defining the environment
        self.x = np.random.randint(1,10000)
        self.y = np.random.randint(1,10000)
        self.TOTAL_TIME_STEPS = 100


        self.dxdy = [(dx, dy) for dx in [-0.1,0,0.1] for dy in [-0.1,0,0.1]]
        # [-0.1, 0, 0.1] for x and [-0.1, 0, 0.1] for y
        # (dx, dy) is [-0.1, 0,0.1]x[-0.1,0,0.1]

        # Define what the agent can do
        # card((dx, dy))=3*3=9
        self.action_space = spaces.Discrete(9)

        self.curr_step = -1


        self.is_optimized = False

        # Observation is the remaining time
        low = np.array([0.0,  # remaining_tries
                        ])
        high = np.array([self.TOTAL_TIME_STEPS,  # remaining_tries
                         ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []


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

        if self.is_optimized:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.is_optimized, {}
        
    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
        #self.action_space.n is max possible actions (ie 9)
        #dxdy= self.dxdy[int(self.action_space.n *random.random())]
        dxdy= self.dxdy[int(action % self.action_space.n) ]
        self.x += dxdy[0]*self.x
        self.y += dxdy[1]*self.y
        if matlab_function(self.x, self.y)==0:
            self.is_optimized=True


        #remaining_steps = self.TOTAL_TIME_STEPS - self.curr_step
        remaining_steps = self.TOTAL_TIME_STEPS - self.curr_step
        time_is_over = (remaining_steps <= 0)
        finish = time_is_over and not self.is_optimized
        if finish:
            self.is_optimized = True  # abuse this a bit


    def _get_reward(self):
        """Reward is given for a sold matlab."""
        if self.is_optimized:
            return 0.0
        else:
            return -matlab_function(self.x,self.y)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.is_optimized = False
        self.x = np.random.randint(1,10000)
        self.y = np.random.randint(1,10000)
        return self._get_state()

    def _render(self, mode='human', close=False):
        pass

    def _get_state(self):
        """Get the observation."""
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob

    def seed(self, seed):
        random.seed(seed)
        np.random.seed()


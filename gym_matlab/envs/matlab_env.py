#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
matlab simulation environment.

Each episode is running a matlab simulation.
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

        # Observations are the current model input 
        low = np.array([0.0,  # remaining_tries
                        ])
        high = np.array([self.TOTAL_TIME_STEPS,  # remaining_tries
                         ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []


    def step(self, action):
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
        if self.is_optimized:
            return 0.0
        else:
            return -matlab_function(self.x,self.y)

    def reset(self):
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
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob

    def seed(self, seed):
        random.seed(seed)
        np.random.seed()

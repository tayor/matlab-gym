#!/usr/bin/env python
# -*- coding: utf-8 -*-

# core modules
import unittest

# 3rd party modules
import gym

# internal modules
import gym_matlab


class Environments(unittest.TestCase):

    def test_env(self):
        env = gym.make('matlab-v0')
        env.seed(0)
        env.reset()
        env.step(0)

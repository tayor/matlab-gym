## [WIP]

This repository contains a PIP package which is an OpenAI environment for doing matlab simulations.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_matlab

env = gym.make('matlab-v0')
```

See https://github.com/matthiasplappert/keras-rl/tree/master/examples for some
examples.
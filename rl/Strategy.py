""" Dynamically choosing learning algorithm """

import sys

from environment import AirHockey
from utils import get_config_strategy

from rl.c51 import c51
from rl.DDQN import DDQN
from rl.DQN import DQN
from rl.DuelingDDQN import DuelingDDQN
from rl.QLearner import QLearner


class Strategy:

    strategies = {
        "q-learner": QLearner,
        "dqn": DQN,
        "ddqn": DDQN,
        "dueling-ddqn": DuelingDDQN,
        "c51": c51,
    }

    def __init__(self):
        pass

    def make(self, name: str, env: AirHockey, agent_name: str = "main"):
        """ Return instance of learner """

        if env is None:
            raise ValueError("Need to pass a gaming environment")

        try:
            config = get_config_strategy(name)
        except KeyError:
            raise KeyError("Strategy not found")

        return self.strategies[name](env, config, agent_name)

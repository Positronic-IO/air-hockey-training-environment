""" Dynamically choosing learning algorithm """

import sys

from environment import AirHockey

from .Agent import Agent
from .agents.A2C import A2C
from .agents.c51 import c51
from .agents.DDQN import DDQN
from .agents.DuelingDDQN import DuelingDDQN
from .agents.QLearner import QLearner
from .utils import get_config_strategy


class Strategy:

    strategies = {"q-learner": QLearner, "ddqn": DDQN, "dueling": DuelingDDQN, "c51": c51, "a2c": A2C}

    def __init__(self):
        pass

    def make(self, env: AirHockey, strategy: str, capacity: int = 0, train: bool = False):
        """ Return instance of learner """

        if strategy == "human":
            return Agent(env)

        config = get_config_strategy(strategy)

        if strategy == "q-learner":
            return self.strategies[strategy](env, config)

        return self.strategies[strategy](env, capacity, train, config)
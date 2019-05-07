""" Dynamically choosing learning algorithm """

import sys

from environment import AirHockey
from rl.Agent import Agent
from rl.c51 import c51
from rl.DDQN import DDQN
from rl.DQN import DQN
from rl.DuelingDDQN import DuelingDDQN
from rl.helpers import TensorBoardLogger
from rl.QLearner import QLearner
from utils import get_config_strategy


class Strategy:

    strategies = {
        "q-learner": QLearner,
        "dqn": DQN,
        "ddqn": DDQN,
        "dueling": DuelingDDQN,
        "c51": c51,
    }

    def __init__(self):
        pass

    def make(
        self, env: AirHockey, strategy: str, capacity: int = 0, train: bool = False
    ):
        """ Return instance of learner """

        if strategy == "human":
            return Agent(env)

        config = get_config_strategy(strategy)

        if strategy == "q-learner":
            return self.strategies[strategy](env, config)

        return self.strategies[strategy](env, capacity, train, config)

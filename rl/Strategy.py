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
        self,
        name: str,
        env: AirHockey,
        tbl: TensorBoardLogger,
        agent_name: str = "main",
    ):
        """ Return instance of learner """

        if env is None:
            raise ValueError("Need to pass a gaming environment")

        if name == "human":
            return Agent(env, "human")

        config = get_config_strategy(name)

        if name in ["dqn", "q-learner"]:
            return self.strategies[name](env, config, agent_name)

        return self.strategies[name](env, config, tbl, agent_name)

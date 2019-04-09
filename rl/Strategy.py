""" Dynamically choosing learning algorithm """

from .q_value import DDQN, DQN, QLearner


class Strategy:

    strategies = {"q-learner": QLearner, "dqn": DQN, "ddqn": DDQN}

    def __init__(self):
        pass

    def make(self, name, env):
        """ Return instance of learner """

        if env is None:
            raise ValueError("Need to pass a gaming environment")

        return self.strategies[name](env)

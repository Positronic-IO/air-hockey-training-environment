""" Dynamically choosing learning algorithm """

from .q_value import DDQN, DQN, DuelingDDQN, QLearner, c51


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

    def make(self, name, env, agent_name="main"):
        """ Return instance of learner """

        if env is None:
            raise ValueError("Need to pass a gaming environment")

        return self.strategies[name](env, agent_name)

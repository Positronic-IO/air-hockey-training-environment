""" Dynamically choosing learning algorithm """

from .q_value import QLearner
from .q_value import DQNLearner
from .q_value import DDQNLearner
from .q_value import C51


class LearnerFactory(object):

    learners = {
        "q-learner": QLearner,
        "dqn": DQNLearner,
        "ddqn": DDQNLearner
    }

    def __init__(self):
        pass

    def make(self, name, env):
        """ Return instance of learner """

        if env is None:
            raise ValueError("Need to pass a gaming environment")

        return self.learners[name](env)

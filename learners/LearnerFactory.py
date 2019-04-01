""" Dynamically choosing learning algorithm """

from .QLearner import QLearner
from .DQNLearner import DQNLearner
from .DDQNLearner import DDQNLearner


class LearnerFactory(object):

    learners = {
        "q-learner": QLearner,
        "dqn-learner": DQNLearner,
        "ddqn-learner": DDQNLearner,
    }

    def __init__(self):
        pass

    def make(self, name, env):
        """ Return instance of learner """

        if env is None:
            raise ValueError("Need to pass a gaming environment")

        return self.learners[name](env)

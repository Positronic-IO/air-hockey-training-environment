""" Dynamically choosing learning algorithm """

from .q_value import QLearner
from .q_value import DQNLearner
from .q_value import DDQNLearner


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

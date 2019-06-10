import pytest

from environment import AirHockey
from rl import A2C, DDQN, Agent, DuelingDDQN, QLearner, Strategy, c51


class TestStrategy:
    def setup(self):
        self.env = AirHockey()
        self.train = True
        self.capacity = 4

    def test_human_agent(self):
        """ Test to see if created agent is human """

        strategy = Strategy().make(self.env, "human")
        assert isinstance(strategy, Agent)

    def test_ddqn_agent(self):
        """ Test to see if created agent is ddqn """

        strategy = Strategy().make(self.env, "ddqn", self.train)
        assert isinstance(strategy, DDQN)

    def test_dueling_agent(self):
        """ Test to see if created agent is dueling """

        strategy = Strategy().make(self.env, "dueling", self.train)
        assert isinstance(strategy, DuelingDDQN)

    def test_c51_agent(self):
        """ Test to see if created agent is c51 """

        strategy = Strategy().make(self.env, "c51", self.train)
        assert isinstance(strategy, c51)

    def test_q_learner_agent(self):
        """ Test to see if created agent is q-learner """

        strategy = Strategy().make(self.env, "q-learner", self.train)
        assert isinstance(strategy, QLearner)

    def test_a2c_agent(self):
        """ Test to see if created agent is A2C """

        strategy = Strategy().make(self.env, "a2c", self.train)
        assert isinstance(strategy, A2C)

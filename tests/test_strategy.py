import pytest

from environment import AirHockey
from rl.Agent import Agent
from rl.c51 import c51
from rl.DDQN import DDQN
from rl.DQN import DQN
from rl.DuelingDDQN import DuelingDDQN
from rl.helpers import TensorBoardLogger
from rl.QLearner import QLearner
from rl.Strategy import Strategy


class TestStrategy:
    def setup(self):
        self.env = AirHockey()
        self.tbl = TensorBoardLogger(log_dir="logs/test")

    def test_human_agent(self):
        """ Test to see if created agent is human """

        strategy = Strategy().make(
            name="human", env=self.env, tbl=self.tbl, agent_name="agent"
        )
        assert isinstance(strategy, Agent)

    def test_ddqn_agent(self):
        """ Test to see if created agent is ddqn """

        strategy = Strategy().make(
            name="ddqn", env=self.env, tbl=self.tbl, agent_name="agent"
        )
        assert isinstance(strategy, DDQN)

    def test_dqn_agent(self):
        """ Test to see if created agent is dqn """

        strategy = Strategy().make(
            name="dqn", env=self.env, tbl=self.tbl, agent_name="agent"
        )
        assert isinstance(strategy, DQN)

    def test_dueling_agent(self):
        """ Test to see if created agent is dueling """

        strategy = Strategy().make(
            name="dueling", env=self.env, tbl=self.tbl, agent_name="agent"
        )
        assert isinstance(strategy, DuelingDDQN)

    def test_c51_agent(self):
        """ Test to see if created agent is c51 """

        strategy = Strategy().make(
            name="c51", env=self.env, tbl=self.tbl, agent_name="agent"
        )
        assert isinstance(strategy, c51)

    def test_q_learner_agent(self):
        """ Test to see if created agent is q-learner """

        strategy = Strategy().make(
            name="q-learner", env=self.env, tbl=self.tbl, agent_name="agent"
        )
        assert isinstance(strategy, QLearner)

    def test_env(self):
        """ Test to see if environment is loaded """

        with pytest.raises(TypeError):
            strategy = Strategy().make(
                name="q-learner", tbl=self.tbl, agent_name="agent"
            )

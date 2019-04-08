""" Q-Learner """

import random
from typing import Tuple

import numpy as np

from environment import Agent
from utils import Observation, State


class QLearner(Agent):
    """ Uses Q-learning to update/maximize rewards """

    def __init__(self, env):
        super().__init__(env)
        self._Q = {}
        self._last_state = None
        self._last_action = None
        self._learning_rate = 0.7
        self._discount = 0.9
        self._epsilon = 0.9
        self._learning = True

    def get_action(self, state: State) -> str:
        """ Give current state, predict next action which maximizes reward """

        # Helps over fitting, encourages to exploration
        if state in self._Q and np.random.uniform(0, 1) < self._epsilon:
            # We use the action which corresponds to the highest Q-value
            action = max(self._Q[state.agent_state], key=self._Q[state.agent_state].get)
        else:
            action = np.random.choice(self.env.actions)
            if state not in self._Q:
                self._Q[state.agent_state] = {}
            self._Q[state.agent_state][action] = 0

        self._last_state = state.agent_state
        self._last_action = action

        return action

    def update(self, data: Observation) -> None:
        """ Updates learner with new state/Q values """

        if self._learning:
            old = self._Q[self._last_state][self._last_action]

            if data.new_state.agent_state in self._Q:
                # Discount reward so we are not too fixated on short-term success.
                # Helps the algorithm focus on the future.
                new = (
                    self._discount
                    * self._Q[data.new_state.agent_state][
                        max(
                            self._Q[data.new_state.agent_state],
                            key=self._Q[data.new_state.agent_state].get,
                        )
                    ]
                )
            else:
                new = 0

            # Update Q-values
            self._Q[self._last_state][self._last_action] = (
                1 - self._learning_rate
            ) * old + self._learning_rate * (data.reward + new)

        return None

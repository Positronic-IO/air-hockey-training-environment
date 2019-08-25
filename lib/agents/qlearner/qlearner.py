""" Q-Learner """
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from lib.types import Action, Observation, State

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QLearner:
    """ Uses Q-learning to update/maximize rewards """

    def __init__(self, env: "AirHockey"):
        self.env = env
        self.Q = dict()
        self.last_state = None
        self.last_action = None
        self.learning_rate = 0.7
        self.gamma = 0.9
        self.epsilon = 0.9
        self.agent_name = ""

    def move(self, action: Action) -> None:
        """ Move agent """

        self.env.update_state(action, self.agent_name)
        return None

    def location(self) -> Union[None, Tuple[int, int]]:
        """ Return agent's location """

        if self.agent_name == "robot":
            return self.env.robot.location()
        elif self.agent_name == "opponent":
            return self.env.opponent.location()

        logging.error("Invalid agent name")
        raise ValueError

    def get_action(self, state: "State") -> int:
        """ Give current state, predict next action which maximizes reward """

        # Helps over fitting, encourages to exploration
        if state in self.Q and np.random.uniform(0, 1) < self.epsilon:
            # We use the action which corresponds to the highest Q-value
            action = max(self.Q[state.agent_location], key=self.Q[state.agent_location].get)
        else:
            action = np.random.randint(0, len(self.env.actions))
            if state not in self.Q:
                self.Q[state.agent_location] = {}
            self.Q[state.agent_location][action] = 0

        self.last_state = state.agent_location
        self.last_action = action

        return action

    def update(self, data: "Observation") -> None:
        """ Updates learner with new state/Q values """

        old = self.Q[self.last_state][self.last_action]

        if data.new_state.agent_location in self.Q:
            # Discount reward so we are not too fixated on short-term success.
            # Helps the algorithm focus on the future.
            new = (
                self.gamma
                * self.Q[data.new_state.agent_location][
                    max(self.Q[data.new_state.agent_location], key=self.Q[data.new_state.agent_location].get)
                ]
            )
        else:
            new = 0

        # Update Q-values
        self.Q[self.last_state][self.last_action] = (1 - self.learning_rate) * old + self.learning_rate * (
            data.reward + new
        )

        return None

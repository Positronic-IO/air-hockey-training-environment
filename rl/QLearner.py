""" Q-Learner """
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from rl.Agent import Agent
from utils import Action, Observation, State

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QLearner:
    """ Uses Q-learning to update/maximize rewards """

    def __init__(self, env: AirHockey, config: Dict[str, Any]):
        self.env = env
        self.Q = dict
        self.last_state = None
        self.last_action = None
        self.learning_rate = config["params"]["learning_rate"]
        self.gamma = config["params"]["gamma"]
        self.epsilon = config["params"]["epsilon"]
        self._agent_name = ""

        self.version = "0.1.0"
        logger.info(f"Strategy defined for {self._agent_name}: {self.__repr__()}")

    def __repr__(self) -> str:
        return f"{self.__str__()} {self.version}"

    def __str__(self) -> str:
        return "Q-Leaner"

    def move(self, action: Action) -> None:
        """ Move agent """

        self.env.update_state(action, self._agent_name)
        return None

    def location(self) -> Union[None, Tuple[int, int]]:
        """ Return agent's location """

        if self.agent_name == "robot":
            return self.env.robot.location()
        elif self.agent_name == "opponent":
            return self.env.opponent.location()
        else:
            logging.error("Invalid agent name")
            raise ValueError

    def get_action(self, state: State) -> int:
        """ Give current state, predict next action which maximizes reward """

        # Helps over fitting, encourages to exploration
        if state in self.Q and np.random.uniform(0, 1) < self.epsilon:
            # We use the action which corresponds to the highest Q-value
            action = max(
                self.Q[state.agent_location], key=self.Q[state.agent_location].get
            )
        else:
            action = np.random.randint(0, len(self.env.actions))
            if state not in self.Q:
                self.Q[state.agent_location] = {}
            self.Q[state.agent_location][action] = 0

        self.last_state = state.agent_location
        self.last_action = action

        return action

    def update(self, data: Observation) -> None:
        """ Updates learner with new state/Q values """

        old = self.Q[self.last_state][self.last_action]

        if data.new_state.agent_location in self.Q:
            # Discount reward so we are not too fixated on short-term success.
            # Helps the algorithm focus on the future.
            new = (
                self.gamma
                * self.Q[data.new_state.agent_location][
                    max(
                        self.Q[data.new_state.agent_location],
                        key=self.Q[data.new_state.agent_location].get,
                    )
                ]
            )
        else:
            new = 0

        # Update Q-values
        self.Q[self.last_state][self.last_action] = (
            1 - self.learning_rate
        ) * old + self.learning_rate * (data.reward + new)

        return None

    @property
    def agent_name(self) -> None:
        return self._agent_name

    @agent_name.setter
    def agent_name(self, name: str) -> None:
        self._agent_name = name

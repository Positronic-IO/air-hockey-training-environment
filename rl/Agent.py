""" Initialize an agent for a game """
import json
import logging
import os
from typing import Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from utils import Action, get_model_path

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Agent:
    def __init__(self, env: AirHockey):

        self.env = env
        self._agent_name = ""

    def move(self, action: Action) -> None:
        """ Move agent """

        action = int(action) if isinstance(action, np.int64) else action
        return self.env.update_state(action=action, agent_name=self._agent_name)

    def location(self) -> Union[None, Tuple[int, int]]:
        """ Return agent's location """

        locations = {
            "robot": self.env.robot.location(),
            "opponent": self.env.opponent.location(),
        }

        if not agents.get(self._agent_name):
            logger.exception("Improper agent name")
            raise ValueError

        return locations.get(self._agent_name)

    @property
    def agent_name(self) -> None:
        return self._agent_name

    @agent_name.setter
    def agent_name(self, name: str) -> None:
        self._agent_name = name

""" Initialize an agent for a game """
import json
import logging
import os
from typing import Dict, Tuple, Union

import numpy as np
from keras.models import Model, load_model

from environment import AirHockey
from rl.helpers import huber_loss
from utils import Action, get_model_path

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Agent:
    def __init__(self, env: AirHockey):

        self.env = env

        self._agent_name = ""
        self.model = Model()
        self.load_path = ""
        self.save_path = ""

    def move(self, action: Action) -> None:
        """ Move agent """

        action = int(action) if isinstance(action, np.int64) else action
        return self.env.update_state(action=action, agent_name=self._agent_name)

    def location(self) -> Union[None, Tuple[int, int]]:
        """ Return agent's location """

        agents = {
            "robot": self.env.robot.location(),
            "opponent": self.env.opponent.location(),
        }

        if not agents.get(self._agent_name):
            logger.exception("Improper agent name")
            raise ValueError

        return agents.get(self._agent_name)

    def load_model(self) -> None:
        """ Load a model"""

        logger.info(f"Saving model from: {self.load_path}")

        self.model = load_model(
            self.load_path, custom_objects={"huber_loss": huber_loss}
        )

    def save_model(self) -> None:
        """ Save a model """

        logger.info(f"Saving model to: {self.save_path}")

        # Create path with epoch number
        head, ext = os.path.splitext(self.save_path)
        path = get_model_path(self.save_path)
        self.model.save(path, overwrite=True)

    @property
    def agent_name(self) -> None:
        return self._agent_name

    @agent_name.setter
    def agent_name(self, name: str) -> None:
        self._agent_name = name

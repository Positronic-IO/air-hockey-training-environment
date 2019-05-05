""" Initialize an agent for a game """
import os
from typing import Tuple, Union

import numpy as np
from keras.models import Model, load_model

from environment import AirHockey
from rl.helpers import huber_loss
from utils import Action, get_model_path


class Agent:
    def __init__(self, env: AirHockey, agent_name: str = "main"):

        self.env = env
        self.agent_name = agent_name
        self.model = Model()
        self._load_path = ""
        self._save_path = ""

    def move(self, action: Action) -> None:
        """ Move agent """

        action = int(action) if isinstance(action, np.int64) else action

        self.env.update_state(agent_name=self.agent_name, action=action)

    def location(self) -> Union[None, Tuple[int, int]]:
        """ Return agent's location """

        if self.agent_name == "main":
            return self.env.agent.location()
        elif self.agent_name == "opponent":
            return self.env.opponent.location()
        else:
            raise ValueError("Invalid agent name")

    def load_model(self, name: str) -> None:
        """ Load a model"""

        print(f"Loading {name} model")

        self.model_path = self._load_path
        self.model = load_model(
            self._load_path, custom_objects={"huber_loss": huber_loss}
        )

    def save_model(self, name: str) -> None:
        """ Save a model """

        print(f"Saving {name} model")

        # If we are not given a path, use the same path as the one we loaded the model
        if not self._save_path:
            self._save_path = self.model_path

        # Create path with epoch number
        head, ext = os.path.splitext(self._save_path)
        path = get_model_path(self._save_path)
        self.model.save(path, overwrite=True)

    @property
    def save_path(self) -> None:
        return self._save_path

    @save_path.setter
    def save_path(self, path: str) -> None:
        self._save_path = path

    @property
    def load_path(self) -> None:
        return self._load_path

    @load_path.setter
    def load_path(self, path: str) -> None:
        self._load_path = path

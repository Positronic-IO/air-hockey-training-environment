""" Initialize an agent for a game """
import json
import os
from typing import Dict, Tuple, Union

import numpy as np
from keras.models import Model, load_model

from environment import AirHockey
from rl.helpers import huber_loss
from utils import Action, get_model_path
from connect import Connection

class Agent:

    # Using test server with Environment
    conn = Connection().make("server",  test=True)

    def __init__(self):

        self._agent_name = ""
        self.model = Model()
        self.load_path = ""
        self.save_path = ""

    def move(self, action: Action) -> None:
        """ Move agent """

        action = int(action) if isinstance(action, np.int64) else action
        self.conn.post({self._agent_name: action})

    def location(self) -> Union[None, Tuple[int, int]]:
        """ Return agent's location """

        return self.conn.get(self._agent_name)[self._agent_name]

    def load_model(self) -> None:
        """ Load a model"""

        print(f"Loading model")

        self.model = load_model(
            self.load_path, custom_objects={"huber_loss": huber_loss}
        )

    def save_model(self) -> None:
        """ Save a model """

        print(f"Saving model")

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
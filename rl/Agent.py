""" Initialize an agent for a game """
import os
from typing import Tuple

from keras.models import Model, load_model

from rl.helpers import huber_loss
from utils import Action, get_model_path


class Agent:
    def __init__(self, env=None):

        if env is not None:
            self.env = env
        else:
            raise ValueError("Please pass an instance of the gaming environment")

        self.model = Model()

    def move(self, action: Action) -> None:
        """ Move agent """

        self.env.update_state(action)
        return None

    def location(self) -> Tuple[int, int]:
        """ Return agent's location """
        return self.env.agent.location()

    def load_model(self, path: str) -> None:
        """ Load a model"""

        print("Loading model")

        self.model_path = path
        self.model = load_model(path, custom_objects={"huber_loss": huber_loss})

    def save_model(self, path: str = "", epoch: int = 0) -> None:
        """ Save a model """
        # If we are not given a path, use the same path as the one we loaded the model
        if not path:
            path = self.model_path

        # Create path with epoch number
        head, ext = os.path.splitext(path)
        path = get_model_path(f"{head}_{epoch}" + ext)
        self.model.save(path)

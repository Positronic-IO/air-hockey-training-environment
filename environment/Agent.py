""" Initialize an agent for a game """
from typing import Tuple
from utils import Action


class Agent(object):
    def __init__(self, env=None):

        if env is not None:
            self.env = env
        else:
            raise ValueError("Please pass an instance of the gaming environment")

    def move(self, action: Action) -> None:
        " Move agent "

        self.env.update_state(action)
        return None

    def location(self) -> Tuple[int, int]:
        return self.env.left_mallet.location()


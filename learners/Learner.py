""" Abstract class for all learners """

from abc import ABC, abstractmethod

from environment import Agent
from typing import Tuple


class Learner(Agent):
    """ Main class for all learners """

    def __init__(self, env):
        super().__init__(env)

    def get_action(self, state: Tuple[int, int]) -> str:
        """ Give current state, predict next action which maximizes reward """

    def update(self, new_state: Tuple[int, int], reward: int) -> None:
        """ Updates learner with new state/Q values """

""" Agent Contract """
import os
from typing import Union

import numpy as np

from environment import AirHockey
from lib.rewards import Rewards
from lib.types import Action, Observation, State
from lib.utils.exceptions import ProjectNotFoundError
from lib.utils.helpers import serialize_state


class Agent:
    def __init__(self, env: "AirHockey", train: bool):
        self.env = env
        self.train = train
        self.reward = 0
        self.done = False
        self.name = "robot"
        self.reward_tracker = Rewards(self.name, self.env.left_goal, self.env.right_goal, self.env.table)
        self.path = os.getenv("PROJECT")

        if not self.path and self.train:
            raise ProjectNotFoundError

    def model_path(self, strategy: str = "") -> str:
        """ Find model path """
        if os.getenv("LOAD_RUN"):  # Use a past run
            return os.path.join("/", "data", "air-hockey", "output", str(os.getenv("LOAD_RUN"))), True
        return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), strategy)), False

    def move(self, action: "Action"):
        """ Move agent """

        action = int(action) if isinstance(action, np.int64) else action
        return self.env.update_state(action=action, agent_name=self.name)

    def _get_action(self, state: np.ndarray) -> Action:
        """ Get action of an agent based on its policy """
        pass

    def get_action(self) -> "Action":
        """ Get the action of an agent """

        # Grab current state of game
        state = self.env.get_state(agent_name=self.name)
        action = self._get_action(serialize_state(state))
        return action

    def save(self):
        """ Save the agent """
        pass

    def update(self, data: Observation):
        pass

    def step(self, action: Action) -> Union[int, "Observation"]:
        """ Next step in the MDP """

        # Grab current state of game
        state = self.env.get_state(agent_name=self.name)

        # Move agent
        self.move(action)

        #  New state of the game
        new_state = self.env.get_state(agent_name=self.name)

        # Compute rewards of round, whether someone score or not, if the episode is over
        mallet = self.env.robot if self.name == "robot" else self.env.opponent
        reward, score, done = self.reward_tracker(self.env.puck, mallet)

        # New observation and have agent learn from it
        observation = Observation(state=state, action=action, reward=reward, done=done, new_state=new_state)

        if self.train:
            self.update(observation)

        # Return useful information
        return score, observation

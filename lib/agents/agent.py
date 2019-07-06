""" Agent Contract """
import numpy as np
from typing import Union

from environment import AirHockey
from lib.types import Action, State, Observation
from lib.rewards import Rewards
from lib.utils.helpers import serialize_state

class Agent:
    def __init__(self, env: "AirHockey"):
        self.env = env
        self.reward = 0
        self.done = False
        self.name = "robot"
        self.reward_tracker = Rewards(self.name, self.env.left_goal, self.env.right_goal, self.env.table)

    def move(self, action: "Action"):
        """ Move agent """

        action = int(action) if isinstance(action, np.int64) else action
        return self.env.update_state(action=action, agent_name=self.name)

    def _get_action(self, state: np.ndarray) -> Action:
        """ Get action of an agent based on its policy """
        raise NotImplementedError

    def get_action(self) -> "Action":
        """ Get the action of an agent """

        # Grab current state of game
        state = self.env.get_state(agent_name=self.name)
        serialized = serialize_state(state)
        action = self._get_action(serialized)
        return action

    def load(self):
        """ Load the agent """
        raise NotImplementedError

    def save(self):
        """ Save the agent """
        raise NotImplementedError

    def build_model(self):
        """ Construct the model(s) """
        raise NotImplementedError

    def update(self, data: Observation):
        raise NotImplementedError

    def step(self, action: Action)  -> Union[int, "Observation"]:
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
        self.update(observation)

        # Return useful information
        return score, observation

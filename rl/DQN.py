""" DQN """

import os
import random
from typing import Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.Networks import Networks
from utils import Observation, State, get_config, get_model_path


class DQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(
        self,
        env: AirHockey,
        config: Dict[str, Dict[str, int]],
        agent_name: str = "main",
    ):
        super().__init__(env, agent_name)

        # Get main config file
        main_config = get_config()

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (3, main_config["capacity"], 2)
        self.action_size = len(self.env.actions)

        # Hyperparameters
        self.gamma = config["params"]["gamma"]
        self.epsilon = config["params"]["epsilon"]

        # Model construction
        self.build_model()

        self.version = "0.1.0"

    def build_model(self) -> None:
        """ Create our DNN model for Q-value approximation """

        self.model = Networks().dqn(self.state_size)
        print(self.model.summary())
        return None

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Compute rewards for any posible action
        rewards = self.model.predict(np.array([state]), batch_size=1)
        assert len(rewards[0]) == self.action_size

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            # We use the action which corresponds to the highest reward
            idx = np.random.randint(0, self.action_size)
        else:
            idx = np.argmax(rewards[0])

        # Update
        self.last_state = state
        self.last_action = idx
        self.last_target = rewards

        return idx

    def update(self, data: Observation) -> None:
        """ Updates learner with new state/Q values """

        rewards = self.model.predict(np.array([data.new_state]), batch_size=1)
        assert len(rewards[0]) == self.action_size

        # Update action we should take, then break out of loop
        self.last_target[0][self.last_action] = (
            data.reward + self.gamma * rewards[0].max()
        )

        # Update model
        self.model.fit(
            np.array([self.last_state]),
            self.last_target,
            batch_size=1,
            epochs=1,
            verbose=0,
        )

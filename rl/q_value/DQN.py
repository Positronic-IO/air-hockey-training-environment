""" DNN Q-Learning Approximator """

import os
import random
from typing import Tuple

import numpy as np

from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.Networks import Networks
from utils import Observation, State, get_model_path


class DQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env):
        super().__init__(env)

        # get size of state and action
        self.state_size = (7, 2)
        self.action_size = len(self.env.actions)

        # Hyperparameters
        self.gamma = 0.1
        self.epsilon = 0.9

        # Model construction
        self.build_model()

        self.version = "0.1.0"

    def build_model(self):
        """ Create our DNN model for Q-value approximation """

        self.model = Networks().dqn(self.state_size)
        print(self.model.summary)
        return None

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Compute rewards for any posible action
        rewards = self.model.predict([np.array([state])], batch_size=1)
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

        reward = data.reward
        rewards = self.model.predict(np.array([data.new_state]), batch_size=1)
        reward += self.gamma * rewards[0].max()

        # Update action we should take, then break out of loop
        for i in range(len(self.env.actions)):
            if self.last_action == self.env.actions[i]:
                self.last_target[0][i] = reward

        # Update model
        self.model.fit(
            np.array([self.last_state]),
            self.last_target,
            batch_size=1,
            nb_epoch=1,
            verbose=0,
        )

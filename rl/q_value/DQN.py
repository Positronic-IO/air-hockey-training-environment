""" DNN Q-Learning Approximator """

import os
import random
from typing import Tuple

import numpy as np
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Dropout, Flatten
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from rl.Agent import Agent
from rl.helpers import huber_loss
from utils import Observation, State, get_model_path


class DQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env):
        super().__init__(env)
        self.gamma = 0.1
        self.epsilon = 0.9

        self.model = self.build_model()

        self.version = "0.1.0"

    def build_model(self):
        """ Create our DNN model for Q-value approximation """

        model = Sequential()

        model.add(Dense(12, kernel_initializer="normal", input_shape=(7, 2)))
        model.add(Activation("relu"))

        model.add(Dense(30, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(20, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Flatten())

        model.add(Dense(4, kernel_initializer="random_uniform"))
        model.add(Activation("linear"))

        rms = (
            RMSprop()
        )  # RMS is used since it is adaptive and our "dataset is not fixed"
        model.compile(loss=huber_loss, optimizer=rms)

        print(model.summary())

        return model

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Compute rewards for any posible action
        rewards = self.model.predict([np.array([state])], batch_size=1)
        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            # We use the action which corresponds to the highest reward
            idx = np.random.randint(0, len(self.env.actions))
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

    def load_model(self, path: str) -> None:
        """ Load a model"""

        self.model_path = path
        self.model.load(path)

    def save_model(self, path: str = "", epoch: int = 0) -> None:
        """ Save a model """
        # If we are not given a path, use the same path as the one we loaded the model
        if not path:
            path = self.model_path

        # Create path with epoch number
        head, ext = os.path.splitext(path)
        path = get_model_path(f"{head}_{epoch}" + ext)
        self.model.save(path)

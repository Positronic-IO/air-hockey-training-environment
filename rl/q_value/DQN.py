""" DNN Q-Learning Approximator """

import os
import random
from typing import Tuple

import numpy as np
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Dropout
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from environment import Agent
from rl.helpers import huber_loss
from utils import Observation, State, get_model_path


class DQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env):
        super().__init__(env)
        self._learning = True
        self._learning_rate = 0.1
        self._gamma = 0.1
        self._epsilon = 0.9

        self._model = self._build_model()

    def _build_model(self):
        """ Create our DNN model for Q-value approximation """

        model = Sequential()

        model.add(Dense(12, kernel_initializer="normal", input_shape=(3, 2)))
        model.add(Activation("relu"))

        model.add(Dense(30, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(20, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(4, kernel_initializer="random_uniform"))
        model.add(Activation("linear"))

        rms = (
            RMSprop()
        )  # RMS is used since it is adaptive and our "dataset is not fixed"
        model.compile(loss=huber_loss, optimizer=rms)

        print(model.summary())

        return model

    def get_action(self, state: State) -> str:
        """ Apply an espilon-greedy policy to pick next action """

        # Compute rewards for any posible action
        rewards = self._model.predict([np.array([state])], batch_size=1)

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self._epsilon:
            # We use the action which corresponds to the highest reward
            idx = np.argmax(rewards[0][0])
            action = self.env.actions[idx]

        else:
            action = np.random.choice(self.env.actions)

        # Update
        self._last_state = state
        self._last_action = action
        self._last_target = rewards

        return action

    def update(self, data: Observation) -> None:
        """ Updates learner with new state/Q values """

        if self._learning:
            reward = data.reward
            rewards = self._model.predict(np.array([data.new_state]), batch_size=1)
            reward += self._gamma * rewards[0][0].max()

            # Update action we should take, then break out of loop
            for i in range(len(self.env.actions)):
                if self._last_action == self.env.actions[i]:
                    self._last_target[0][0][i] = reward
                    break

            # Update model
            self._model.fit(
                np.array([self._last_state]),
                self._last_target,
                batch_size=1,
                nb_epoch=1,
                verbose=0,
            )

        return None

    def load_model(self, path: str) -> None:
        """ Load a model"""

        self.model_path = path
        self._model.load_weights(path)

    def save_model(self, path: str = "", epoch: int = 0) -> None:
        """ Save a model """
        # If we are not given a path, use the same path as the one we loaded the model
        if not path:
            path = self.model_path

        # Create path with epoch number
        head, ext = os.path.splitext(path)
        path = get_model_path(f"{head}_{epoch}" + ext)
        self._model.save_weights(path)

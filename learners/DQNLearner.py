""" DNN Q-Learning Approximator """

import os
import random
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from utils import get_model_path
from .Learner import Learner


class DQNLearner(Learner):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env):
        super().__init__(env)
        self._learning = True
        self._learning_rate = 0.1
        self._discount = 0.1
        self._epsilon = 0.9

        self._model = self._build_model()

    def _huber_loss(self, y_true, y_pred, clip_delta: float = 1.0) -> float:
        """ Compute Huber Loss 
        
        References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
        """

        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (
            K.abs(error) - clip_delta
        )

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        """ Create our DNN model for Q-value approximation """

        model = Sequential()

        model.add(Dense(4, kernel_initializer="normal", input_shape=(2,2)))
        model.add(Activation("relu"))

        model.add(Dense(20, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(30, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(30, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(10, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(4, kernel_initializer="random_uniform"))
        model.add(Activation("linear"))

        rms = (
            RMSprop()
        )  # RMS is used since it is adaptive and our "dataset is not fixed"
        model.compile(loss=self._huber_loss, optimizer=rms)

        print(model.summary())

        return model

    def get_action(self, state: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        """ Give current state, predict next action which maximizes reward """

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

    def update(
        self, new_state: Tuple[Tuple[int, int], Tuple[int, int]], reward: int
    ) -> None:
        """ Updates learner with new state/Q values """

        if self._learning:
            rewards = self._model.predict(np.array([new_state]), batch_size=1)
            maxQ = rewards[0].max()
            new = self._discount * maxQ

            if self._last_action == self.env.actions[0]:
                self._last_target[0][0][0] = reward + new

            if self._last_action == self.env.actions[1]:
                self._last_target[0][0][1] = reward + new

            if self._last_action == self.env.actions[2]:
                self._last_target[0][0][2] = reward + new

            if self._last_action == self.env.actions[3]:
                self._last_target[0][0][3] = reward + new

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

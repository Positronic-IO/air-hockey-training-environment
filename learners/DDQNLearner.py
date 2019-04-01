""" DNN Q-Learning Approximator """

import os
import random
from collections import deque
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Dropout
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from utils import get_model_path

from .Learner import Learner


class DDQNLearner(Learner):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env):
        super().__init__(env)
        # create replay memory using deque
        self.memory = list()
        self.max_memory = 50000  # number of previous transitions to remember

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 500
        self._model = self._build_model()

    def _huber_loss(
        self, y_true: float, y_pred: float, clip_delta: float = 1.0
    ) -> float:
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

        model.add(Dense(4, kernel_initializer="normal", input_shape=(2, 2)))
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

    def remember(
        self,
        state: Tuple[Tuple[int, int], Tuple[int, int]],
        action: str,
        reward: int,
        next_state: Tuple[Tuple[int, int], Tuple[int, int]],
    ):
        """ Push data into memory for replay later """
        self.memory.append((state, action, reward, next_state))

    def get_action(self, state: Tuple[Tuple[int, int], Tuple[int, int]]) -> str:
        """ Give current state, predict next action which maximizes reward """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.env.actions)

        # Compute rewards for any posible action
        rewards = self._model.predict([np.array([state])], batch_size=1)
        idx = np.argmax(rewards[0][0])
        return self.env.actions[idx]

    def update(self) -> None:
        """ Experience replay """

        # Update model in intervals
        if len(self.memory) % self.batch_size == 0:

            if len(self.memory) > self.max_memory:
                self.memory.pop()

            print("Updating replay")

            minibatch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state in minibatch:
                target = self._model.predict(np.array([next_state]))

                if action == self.env.actions[0]:
                    target[0][0][0] = reward + self.gamma * target[0].max()

                if action == self.env.actions[1]:
                    target[0][0][1] = reward + self.gamma * target[0].max()

                if action == self.env.actions[2]:
                    target[0][0][2] = reward + self.gamma * target[0].max()

                if action == self.env.actions[3]:
                    target[0][0][3] = reward + self.gamma * target[0].max()

                self._model.fit(np.array([state]), target, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

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

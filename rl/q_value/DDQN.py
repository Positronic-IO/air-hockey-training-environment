""" DNN Q-Learning Approximator """

import os
import random
import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Dropout
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from environment import Agent
from rl.helpers import huber_loss
from utils import Observation, State, get_model_path


class DDQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env):
        super().__init__(env)
        # Replay memory
        self.memory = list()
        self.max_memory = 10 ** 5  # number of previous transitions to remember

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 10 ** 2
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

        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))

        print(model.summary())

        return model

    def remember(self, state: State, action: str, reward: int, next_state: State):
        """ Push data into memory for replay later """
        self.memory.append((state, action, reward, next_state))

    def get_action(self, state: State) -> str:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.env.actions)

        # Compute rewards for any posible action
        rewards = self._model.predict([np.array([state])], batch_size=1)
        idx = np.argmax(rewards[0][0])
        return self.env.actions[idx]

    def update(self, data: Observation, iterations: int) -> None:
        """ Experience replay """

        # Push data into observat
        self.memory.append(data)

        # Update model in intervals
        if iterations > self.batch_size and iterations % self.batch_size == 0:

            # Governs how much history is stored in memory
            if len(self.memory) > self.max_memory:
                self.memory.pop()

            print("Updating replay")
            # Sample observations from memory for experience replay
            minibatch = random.sample(self.memory, self.batch_size)
            for observation in minibatch:
                reward = observation.reward
                target = self._model.predict(np.array([observation.new_state]))
                reward += self.gamma * target[0][0].max()

                # Update action we should take, then break out of loop
                for i in range(len(self.env.actions)):
                    if observation.action == self.env.actions[i]:
                        target[0][0][i] = reward
                        break

                self._model.fit(
                    np.array([observation.state]), target, epochs=1, verbose=0
                )

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return None

    def load_model(self, path: str) -> None:
        """ Load a model"""

        self.model_path = path
        self._model.load_weights(path)
        print("Model loaded")

    def save_model(self, path: str = "", epoch: int = 0) -> None:
        """ Save a model """
        # If we are not given a path, use the same path as the one we loaded the model
        if not path:
            path = self.model_path

        # Create path with epoch number
        head, ext = os.path.splitext(path)
        path = get_model_path(f"{head}_{epoch}" + ext)
        self._model.save_weights(path)

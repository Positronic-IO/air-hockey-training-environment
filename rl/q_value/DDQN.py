""" DNN Q-Learning Approximator """

import os
import random
import time
from typing import Tuple
from collections import deque

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Dropout, Flatten
from keras.layers.core import Activation, Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop

from rl.Agent import Agent
from rl.helpers import huber_loss
from utils import Observation, State, get_model_path


class DDQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env):
        super().__init__(env)
        # Replay memory
        self.memory = deque(maxlen=self.max_memory)
        self.max_memory = 10 ** 7  # number of previous transitions to remember

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 10 ** 3
        self.sync_target_interval = 10 ** 5

        self.target_model = self.build_model()
        self.model = self.build_model()

        self.batch_counter = 0

        self.version = "0.3.0"

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
        model.add(Activation("linear    "))

        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))

        print(model.summary())

        return model

    def update_target_model(self) -> None:
        """ Copy weights from model to target_model """

        print("Sync target model")
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, data: Observation):
        """ Push data into memory for replay later """

        # Push data into observation and remove one from buffer
        self.memory.append(data)
        self.memory.popleft()

    def get_action(self, state: State) -> str:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.env.actions)

        # Compute rewards for any posible action
        rewards = self.model.predict([np.array([state])], batch_size=1)
        idx = np.argmax(rewards[0])
        return self.env.actions[idx]

    def update(self, data: Observation) -> None:
        """ Experience replay """

        # Push data into observation and remove one from buffer
        self.remember(data)

        assert len(self.memory) < self.max_memory + 1, "Max memory exceeded"

        # Update model in intervals
        self.batch_counter += 1
        if self.batch_counter > self.sync_target_interval:

            # Reset Batch counter
            self.batch_counter = 0

            print("Updating replay")
            # Sample observations from memory for experience replay
            minibatch = random.sample(self.memory, self.batch_size)
            for observation in minibatch:
                target = self.model.predict(np.array([observation.new_state]))

                if observation.done:
                    # Sync Target Model
                    self.update_target_model()

                    # Update action we should take, then break out of loop
                    for i in range(len(self.env.actions)):
                        if observation.action == self.env.actions[i]:
                            target[0][i] = observation.reward

                else:
                    t = self.target_model.predict(np.array([observation.new_state]))

                    # Update action we should take, then break out of loop
                    for i in range(len(self.env.actions)):
                        if observation.action == self.env.actions[i]:
                            target[0][i] = observation.reward + self.gamma * np.amax(
                                t[0]
                            )

                self.model.fit(
                    np.array([observation.state]), target, epochs=1, verbose=0
                )

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return None

    def load_model(self, path: str) -> None:
        """ Load a model"""

        print("Loading model")

        self.model_path = path
        self.model = load_model(path, custom_objects={"huber_loss": huber_loss})

    def save_model(self, path: str = "", epoch: int = 0) -> None:
        """ Save a model """
        # If we are not given a path, use the same path as the one we loaded the model
        if not path:
            path = self.model_path

        # Create path with epoch number
        head, ext = os.path.splitext(path)
        path = get_model_path(f"{head}_{epoch}" + ext)
        self.model.save(path)

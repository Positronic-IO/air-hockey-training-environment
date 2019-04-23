""" DNN Q-Learning Approximator """

import os
import random
import time
from collections import deque
from typing import Tuple

import numpy as np

from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.Networks import Networks
from utils import Observation, State, get_model_path


class DDQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env, agent_name="main"):
        super().__init__(env, agent_name)

        # get size of state and action
        self.state_size = (7, 2)
        self.action_size = len(self.env.actions)

        # Replay memory
        self.max_memory = 10 ** 7  # number of previous transitions to remember
        self.memory = deque()

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.batch_size = 10 ** 3
        self.sync_target_interval = 10 ** 5

        # Model construction
        self.build_model()

        self.batch_counter = 0

        self.version = "0.3.0"

    def build_model(self) -> None:
        """ Create our DNN model for Q-value approximation """

        model = Networks().ddqn(self.state_size, self.learning_rate)

        self.model = model
        self.target_model = model
        print(self.model.summary())
        return None

    def update_target_model(self) -> None:
        """ Copy weights from model to target_model """

        print("Sync target model")
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, data: Observation) -> None:
        """ Push data into memory for replay later """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        if len(self.memory) > self.max_memory:
            self.memory.popleft()

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)

        # Compute rewards for any posible action
        rewards = self.model.predict([np.array([state])], batch_size=1)
        idx = np.argmax(rewards[0])
        return idx

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

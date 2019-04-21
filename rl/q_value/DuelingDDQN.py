""" Dueling DDQN """

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


class DuelingDDQN(Agent):

    """ Reference: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/dueling_ddqn.py """

    def __init__(self, env):
        super().__init__(env)

        # Replay memory
        self.memory = list()
        self.max_memory = 10 ** 5  # number of previous transitions to remember

        # get size of state and action
        self.state_size = (7, 2)
        self.action_size = len(self.env.actions)

        # these is hyper parameters for the Double DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 0.0001
        self.batch_size = 10 ** 3
        self.observe = 5000
        self.explore = 50000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = (
            10 ** 4
        )  # Number of timesteps between training interval

        # create replay memory using deque
        self.memory = deque()
        self.max_memory = 50000  # number of previous transitions to remember

        # Model construction
        self.build_model()

        # Counters
        self.batch_counter = 0
        self.sync_counter = 0
        self.t = 0

        self.version = "0.1.0"

    def build_model(self):
        """ Create our DNN model for Q-value approximation """

        model = Networks().dueling_ddqn(self.state_size, self.learning_rate)

        self.model = model
        self.target_model = model
        print(self.model.summary())
        return None

    def update_target_model(self) -> None:
        """ After some time interval update the target model to be same with model """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)

        # Compute rewards for any posible action
        rewards = self.model.predict([np.array([state])], batch_size=1)
        idx = np.argmax(rewards[0])
        return idx

    def remember(self, data: Observation) -> None:
        """ Push data into memory for replay later """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Modify epsilon
        if self.epsilon > self.final_epsilon and self.t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        self.t += 1

        # Memory management
        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        assert len(self.memory) < self.max_memory + 1, "Max memory exceeded"

    def update(self, data: Observation) -> None:
        """ Experience replay """

        # Push data into observation and remove one from buffer
        self.remember(data)

        # Update the target model to be same with model
        self.sync_counter += 1
        if self.sync_counter > self.update_target_freq:
            # Sync Target Model
            self.update_target_model()
            self.sync_counter = 0

        # Update model in intervals
        self.batch_counter += 1
        if self.batch_counter > self.timestep_per_train:
            self.batch_counter = 0

            print("Update Model")

            num_samples = min(
                self.batch_size * self.timestep_per_train, len(self.memory)
            )
            replay_samples = random.sample(self.memory, num_samples)

            update_input = np.zeros(((num_samples,) + self.state_size))
            update_target = np.zeros(((num_samples,) + self.state_size))
            action, reward, done = list(), list(), list()

            for i in range(num_samples):
                update_input[i, :, :] = replay_samples[i][0]
                action.append(replay_samples[i][1])
                reward.append(replay_samples[i][2])
                done.append(replay_samples[i][3])
                update_target[i, :, :] = replay_samples[i][4]

            target = self.model.predict(update_input)
            target_val = self.model.predict(update_target)
            target_val_ = self.target_model.predict(update_target)
            print(target.shape)
            print(target_val.shape)
            print(target_val_.shape)
            for i in range(num_samples):
                # like Q Learning, get maximum Q value at s'
                # But from target model
                if done[i]:
                    target[i][action[i]] = reward[i]
                else:
                    # the key point of Double DQN
                    # selection of action is from model
                    # update is from target model
                    a = np.argmax(target_val[i])
                    target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])
            loss = self.model.fit(
                update_input, target, batch_size=self.batch_size, epochs=1, verbose=0
            )

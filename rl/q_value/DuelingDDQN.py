""" Dueling DDQN """

import os
import random
import time
from typing import Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout, Input, Lambda, add, Flatten, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

from rl.Agent import Agent
from rl.helpers import huber_loss
from utils import Observation, State, get_model_path


class DuelingDDQN(Agent):

    """ Reference: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/dueling_ddqn.py """

    def __init__(self, env):
        super().__init__(env)
        # Replay memory
        self.memory = list()
        self.max_memory = 10 ** 5  # number of previous transitions to remember

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
        self.timestep_per_train = 100  # Number of timesteps between training interval

        # create replay memory using deque
        self.memory = list()
        self.max_memory = 50000  # number of previous transitions to remember

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.version = "0.1.0"

    def build_model(self):
        """ Create our DNN model for Q-value approximation """

        state_input = Input(shape=((3, 2)))
        x = Dense(12, kernel_initializer="normal", activation="relu")(state_input)
        x = Dense(30, kernel_initializer="normal", activation="relu")(x)
        x = Dense(20, kernel_initializer="normal", activation="relu")(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, kernel_initializer="normal", activation="relu")(x)
        state_value = Dense(4, kernel_initializer="random_uniform")(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(4,))(
            state_value
        )

        # action advantage tower - A
        action_advantage = Dense(256, kernel_initializer="normal", activation="relu")(x)
        action_advantage = Dense(4)(action_advantage)
        action_advantage = Lambda(
            lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(4,)
        )(action_advantage)

        # merge to state-action value function Q
        state_action_value = add([state_value, action_advantage])

        model.compile(loss=huber_loss, optimizer=Adam(lr=self.learning_rate))

        print(model.summary())

        return model

    def update_target_model(self) -> None:
        """ After some time interval update the target model to be same with model """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, len(self.env.actions))

        # Compute rewards for any posible action
        rewards = self.model.predict([np.array([state])], batch_size=1)
        idx = np.argmax(rewards[0][0])
        return idx

    def update(self, data: Observation, iterations: int) -> None:
        """ Experience replay """

        self.memory.append(tuple(data))
        if self.epsilon > self.final_epsilon and iterations > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        # Governs how much history is stored in memory
        if len(self.memory) > self.max_memory:
            self.memory.pop()

        # Update the target model to be same with model
        if iterations % self.update_target_freq == 0:
            self.update_target_model()

        num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
        replay_samples = random.sample(self.memory, num_samples)

        update_input = np.zeros(((num_samples,) + (3, 2)))
        update_target = np.zeros(((num_samples,) + (3, 2)))
        action, reward = list(), list()

        for i in range(num_samples):
            update_input[i, :, :] = replay_samples[i][0]
            action.append(replay_samples[i][1])
            reward.append(replay_samples[i][2])
            update_target[i, :, :] = replay_samples[i][3]

        target = self.model.predict(update_input)
        target_val = self.model.predict(update_target)
        target_val_ = self.target_model.predict(update_target)
        for i in range(num_samples):
            # the key point of Double DQN
            # selection of action is from model
            # update is from target model
            a = np.argmax(target_val[i][0])
            target[i][self.env.actions.index(action[i])] = reward[i] + self.gamma * (
                target_val_[i][a]
            )
        loss = self.model.fit(
            update_input, target, batch_size=self.batch_size, nb_epoch=1, verbose=0
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

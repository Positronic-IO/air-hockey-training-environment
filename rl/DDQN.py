""" DDQN """

import os
import random
import time
from collections import deque
from typing import Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from rl.Agent import Agent
from rl.helpers import TensorBoardLogger, huber_loss
from rl.MemoryBuffer import MemoryBuffer
from rl.Networks import Networks
from utils import Observation, State, get_config, get_model_path


class DDQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(
        self,
        env: AirHockey,
        config: Dict[str, Dict[str, int]],
        tbl: TensorBoardLogger,
        agent_name: str = "main",
    ):
        super().__init__(env, agent_name)

        # Get main config file
        main_config = get_config()

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (3, main_config["capacity"], 2)
        self.action_size = len(self.env.actions)

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        self.gamma = config["params"]["gamma"]  # discount rate
        self.epsilon = config["params"]["epsilon"]  # exploration rate
        self.epsilon_min = config["params"]["epsilon_min"]
        self.epsilon_decay = config["params"]["epsilon_decay"]
        self.learning_rate = config["params"]["learning_rate"]
        self.batch_size = config["params"]["batch_size"]
        self.sync_target_interval = config["params"]["sync_target_interval"]

        # If we are not training, set our epsilon to final_epsilon.
        # We want to choose our prediction more than a random policy.
        self.train = main_config["train"]
        self.epsilon = self.epsilon if self.train else self.epsilon_min

        # Model construction
        self.build_model()

        # Counters
        self.t = 0

        # Initiate Tensorboard
        self.tbl = tbl

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

    def _epsilon(self) -> None:
        """ Update all things epsilon """

        if not self.train:
            return None

        self.tbl.log_scalar(
            f"{self.__class__.__name__.title()} epsilon", self.epsilon, self.t
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return None

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            idx = np.random.randint(0, self.action_size)
            self.tbl.log_histogram(
                f"{self.__class__.__name__.title()} Greedy Actions", idx, self.t
            )
            return idx

        # Compute rewards for any posible action
        rewards = self.model.predict(np.array([state]), batch_size=1)[0]
        assert len(rewards) == self.action_size

        idx = np.argmax(rewards)
        self.tbl.log_histogram(
            f"{self.__class__.__name__.title()} Predict Actions", idx, self.t
        )
        return idx

    def update(self, data: Observation) -> None:
        """ Experience replay """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Modify epsilon
        self._epsilon()

        # Update model in intervals
        if self.t > self.sync_target_interval:

            print("Updating replay")

            # Sample observations from memory for experience replay
            minibatch = self.memory.sample(self.batch_size)
            for observation in minibatch:
                target = self.model.predict(np.array([observation.new_state]))

                if observation.done:
                    # Sync Target Model
                    self.update_target_model()

                    # Update action we should take, then break out of loop
                    # ! Deprecate
                    for i in range(len(self.env.actions)):
                        if observation.action == self.env.actions[i]:
                            target_[0][i] = observation.reward

                    target[0][observation.action] = observation.reward

                else:
                    t = self.target_model.predict(np.array([observation.new_state]))

                    # Update action we should take
                    target[0][
                        observation.action
                    ] = observation.reward + self.gamma * np.argmax(t[0])

                self.model.fit(
                    np.array([observation.state]), target, epochs=1, verbose=0
                )

           self.t += 1 

        return None

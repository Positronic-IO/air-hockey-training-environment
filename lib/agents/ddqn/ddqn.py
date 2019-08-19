""" DDQN """
import logging
import os
from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from lib.agents import Agent
from lib.agents.ddqn import model
from lib.agents.ddqn.config import config
from lib.buffer import MemoryBuffer
from lib.exploration import EpsilonGreedy
from lib.types import Observation, State
from lib.utils.helpers import serialize_state

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DDQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(self, env: "AirHockey", train: bool, **kwargs):
        super().__init__(env)

        logger.info(f"Strategy defined for {self.name}: {self.__repr__()}")

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 4

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)
        self.gamma = config["params"]["gamma"]  # discount rate
        self.learning_rate = config["params"]["learning_rate"]
        self.batch_size = config["params"]["batch_size"]
        self.sync_target_interval = config["params"]["sync_target_interval"]
        self.timestep_per_train = config["params"]["timestep_per_train"]

        # Are we training?
        self.train = train

        # Model load and save paths
        self.load_path = config.get("load")
        self.save_path = None

        # Model construction
        self.build_model()

        # Counters
        self.t = 0

        # Exploration strategies
        self.exploration_strategy = EpsilonGreedy(action_size=self.action_size)

    def __repr__(self) -> str:
        return "DDQN"

    def build_model(self) -> None:
        """ Create our DNN model for Q-value approximation """

        self.model = model.create(self.state_size, self.learning_rate)
        self.target_model = model.create(self.state_size, self.learning_rate)

        if self.load_path:
            self.load_model()

        # Set up target model
        self.target_model.set_weights(self.model.get_weights())

        print(self.model.summary())
        return None

    def update_target_model(self) -> None:
        """ Copy weights from model to target_model """

        logger.info("Sync target model for DDQN")
        self.target_model.set_weights(self.model.get_weights())

    def _get_action(self, state: "State") -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Compute rewards for any posible action
        q_values = self.model.predict(serialize_state(state)).flatten()
        assert q_values.shape == (self.action_size,), f"Q-values with shape {q_values.shape} have the wrong dimensions"
        return self.exploration_strategy.step(q_values) if self.train else np.argmax(q_values)

    def update(self, data: "Observation") -> None:
        """ Update our model using relay """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Update model in intervals
        if self.t > 0 and self.t % self.timestep_per_train == 0:

            logger.info(f"Updating DDQN model")

            # Sample observations from memory for experience replay
            num_samples = min(self.batch_size, len(self.memory))
            minibatch = self.memory.sample(num_samples)
            for observation in minibatch:
                flattend_state = serialize_state(observation.state)
                flattend_new_state = serialize_state(observation.new_state)
                target = self.model.predict(flattend_new_state).flatten()
                assert target.shape == (
                    self.action_size,
                ), f"Q-values with shape {target.shape} have the wrong dimensions"
                if observation.done:
                    # Sync Target Model
                    self.update_target_model()

                    # Update action we should take, then break out of loop
                    target[observation.action] = observation.reward
                else:
                    t = self.target_model.predict(flattend_new_state).flatten()
                    assert t.shape == (self.action_size,), f"Q-values with shape {t.shape} have the wrong dimensions"
                    # Update action we should take
                    target[observation.action] = observation.reward + self.gamma * np.argmax(t)

                self.model.fit(
                    flattend_state, np.expand_dims(target, axis=0), batch_size=self.batch_size, epochs=1, verbose=0
                )

        # Save model
        if self.train and self.t % self.timestep_per_train == 0:
            self.save_model()

        self.t += 1

        return None

    def load_model(self) -> None:
        """ Load a model"""

        logger.info(f"Loading model from: {self.load_path}")

        self.model.load_weights(self.load_path)

    def save_model(self) -> None:
        """ Save a model's weights """

        # Create path with epoch number
        path = os.path.join(self.save_path, "model.h5")
        logger.info(f"Saving model to: {self.save_path}")
        self.model.save_weights(path, overwrite=True)

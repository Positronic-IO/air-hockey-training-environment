""" DDQN """
import logging
import os
from typing import Any, Dict, Tuple, Union

import numpy as np
from keras.models import load_model

from environment import AirHockey
from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.MemoryBuffer import MemoryBuffer
from rl import networks
from utils import Observation, State

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DDQN(Agent):

    """ Reference: https://keon.io/deep-q-learning/ """

    def __init__(
        self, env: AirHockey, capacity: int, train: bool, config: Dict[str, Any]
    ):
        super().__init__(env)

        self.version = "0.3.0"
        logger.info(f"Strategy defined for {self.agent_name}: {self.__repr__()}")

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 4

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
        self.iterations_on_save = config["params"]["iterations_on_save"]

        # If we are not training, set our epsilon to final_epsilon.
        # We want to choose our prediction more than a random policy.
        self.train = train
        self.epsilon = self.epsilon if self.train else self.epsilon_min

        # Model load and save paths
        self.load_path = None if not config.get("load") else config.get("load")
        self.save_path = None

        # Model construction
        self.build_model()

        # Counters
        self.t = 0

    def __repr__(self) -> str:
        return f"{self.__str__()} {self.version}"

    def __str__(self) -> str:
        return "DDQN"

    def build_model(self) -> None:
        """ Create our DNN model for Q-value approximation """

        model = networks.ddqn(self.state_size, self.learning_rate)

        self.model = model
        self.target_model = model

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

    def _epsilon(self) -> None:
        """ Update all things epsilon """

        if not self.train:
            return None

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return None

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            idx = np.random.randint(0, self.action_size)
            return idx

        # Compute rewards for any posible action
        flattened_state = np.hstack(state)
        rewards = self.model.predict(np.array([flattened_state]), batch_size=1)[0]
        assert len(rewards) == self.action_size

        idx = np.argmax(rewards)
        return idx

    def update(self, data: Observation) -> None:
        """ Update our model using relay """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Modify epsilon
        self._epsilon()

        # Update model in intervals
        if self.t > 0 and self.t % self.sync_target_interval == 0:

            logger.info(f"Updating DDQN model")

            # Sample observations from memory for experience replay
            minibatch = self.memory.sample(self.batch_size)
            for observation in minibatch:
                target = self.model.predict(np.array([observation.new_state]))

                if observation.done:
                    # Sync Target Model
                    self.update_target_model()

                    # Update action we should take, then break out of loop
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

        # Save model
        if self.t % self.iterations_on_save == 0:
            self.save_model()

        self.t += 1

        return None

    def load_model(self) -> None:
        """ Load a model"""

        logger.info(f"Loading model from: {self.load_path}")

        self.model = load_model(
            self.load_path, custom_objects={"huber_loss": huber_loss}
        )

    def save_model(self) -> None:
        """ Save a model """

        # Create path with epoch number
        path = os.path.join(self.save_path, "ddqn.h5")
        logger.info(f"Saving model to: {self.save_path}")
        self.model.save(path, overwrite=True)

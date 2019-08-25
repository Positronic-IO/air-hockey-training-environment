""" PPO """
import imp
import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from environment import AirHockey
from lib.agents import Agent
from lib.buffer import MemoryBuffer
from lib.types import Observation, State
from lib.exploration import SoftmaxPolicy, GaussianWhiteNoiseProcess, OrnsteinUhlenbeckProcess
from lib.utils.helpers import serialize_state


# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PPO(Agent):

    """ Reference: https://github.com/LuEE-C/PPO-Keras """

    def __init__(self, env: "AirHockey", train: bool):
        super().__init__(env)

        # Are we training?
        self.train = train

        # Load raw model
        path, to_load = self.model_path("ppo")
        model = imp.load_source("ppo", os.path.join(path, "model.py"))
        config = model.config()

        # Are we doing continuous or discrete PPO?
        self.continuous = config["continuous"]

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 2 if self.continuous else 4

        # These are hyper parameters for the Policy Gradient
        self.gamma = config["params"]["gamma"]
        self.actor_learning_rate = config["params"]["actor_learning_rate"]
        self.critic_learning_rate = config["params"]["critic_learning_rate"]
        self.batch_size = config["params"]["batch_size"]

        # Train and Save
        self.timestep_per_train = config["params"]["timestep_per_train"]
        self.iterations_on_save = config["params"]["iterations_on_save"]

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)
        self.batch_size = config["params"]["batch_size"]

        # Training epochs
        self.epochs = config["params"]["epochs"]

        self.actor_model, self.critic_model = model.create(
            state_size=self.state_size,
            action_size=self.action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            continuous=self.continuous,
        )

        if to_load:
            try:
                logger.info(f"Loading model's weights from: {path}...")
                self.actor_model.load_weights(os.path.join(path, "actor.h5"))
                self.critic_model.load_weights(os.path.join(path, "critic.h5"))
            except OSError:
                logger.info("Weights file corrupted, starting fresh...")
                pass  # If file is corrupted, move on.

        logger.info("Actor Model")
        logger.info(self.actor_model.summary())
        logger.info("Critic Model")
        logger.info(self.critic_model.summary())

        # Noise (Continuous)
        self.noise = config["params"]["noise"]

        # Keep up with the iterations
        self.t = 0

        # Initialize
        self.action_matrix, self.policy = None, None

        # Exploration and Noise Strategy
        self.exploration_strategy = SoftmaxPolicy(action_size=self.action_size)
        self.noise_strategy = GaussianWhiteNoiseProcess(size=self.action_size)

        logger.info(f"Strategy defined for {self.name}: {self.__repr__()}")

    def __repr__(self):
        return f"{self.__class__.__name__} Continuous" if self.continuous else self.__class__.__name__

    def _get_action(self, state: "State") -> Union[int, Tuple[int, int]]:
        """ Return a action """

        if not self.continuous:
            return self._get_action_discrete(state)

        return self._get_action_continuous(state)

    def _get_action_discrete(self, state: "State") -> int:
        """ Return a random action (discrete space) """
        q_values = self.actor_model.predict(
            [serialize_state(state), np.zeros(shape=(1, 1)), np.zeros(shape=(1, self.action_size))]
        ).flatten()

        # Sanity check
        assert q_values.shape == (
            self.action_size,
        ), f"Policy is of shape {q_values.shape} instead of {(self.action_size,)}"

        action = self.exploration_strategy.step(q_values) if self.train else np.argmax(q_values)
        action_matrix = np.zeros(self.action_size)
        action_matrix[action] = 1
        self.action_matrix, self.q_values = action_matrix, q_values
        return action

    def _get_action_continuous(self, state: "State") -> Tuple[int, int]:
        """ Return a random action (continuous space) """
        policy = self.actor_model.predict(
            [serialize_state(state), np.zeros(shape=(1, 1)), np.zeros(shape=(1, self.action_size))]
        ).flatten()

        if self.train:
            action = action_matrix = policy + self.noise_strategy.sample()
        else:
            action = action_matrix = policy
        self.action_matrix, self.policy = action_matrix, policy
        return action.tolist()

    def discount_rewards(self, rewards: List[float]):
        """ Compute discount rewards """
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * self.gamma
        return rewards

    def update(self, data: "Observation") -> None:
        """ Update policy network every episode """
        self.memory.append((data, (self.action_matrix, self.policy)))

        # Update model in intervals
        if self.t > 0 and self.t % self.timestep_per_train == 0:

            logger.info("Update models")

            observations = self.memory.retreive()
            states = np.array([serialize_state(observation[0].state, dim=1) for observation in observations])
            rewards = np.array([observation[0].reward for observation in observations])
            action_matrices = np.vstack([observation[1][0] for observation in observations])
            policies = np.vstack([observation[1][1] for observation in observations])

            transformed_rewards = np.array(self.discount_rewards(rewards))
            advantages = transformed_rewards - np.array(self.critic_model.predict(states)[0])

            # Train models
            self.actor_model.fit(
                [states, advantages.T, policies],
                [action_matrices],
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=False,
            )
            self.critic_model.fit(
                [states], [transformed_rewards], batch_size=self.batch_size, epochs=self.epochs, verbose=False
            )

            # Empty buffer (treat as a cache for the minibatch)
            self.memory.purge()

        # Save model
        if self.train and self.t % self.timestep_per_train == 0:
            self.save()

        self.t += 1

        return None

    def save(self) -> None:
        """ Save a models """
        logger.info(f"Saving model to: {self.path}")

        # Save actor model
        actor_path = os.path.join(self.path, "actor.h5")
        self.actor_model.save_weights(actor_path, overwrite=True)

        # Save critic model
        critic_path = os.path.join(self.path, "critic.h5")
        self.critic_model.save_weights(critic_path, overwrite=True)

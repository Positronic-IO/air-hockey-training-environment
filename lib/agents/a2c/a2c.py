""" Actor/Critic (Synchronous) """
import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from environment import AirHockey
from lib.agents import Agent
from lib.agents.a2c import model
from lib.agents.a2c.config import config
from lib.buffer import MemoryBuffer
from lib.types import Observation, State
from lib.exploration import SoftmaxPolicy
from lib.utils.helpers import serialize_state

# Set random seeds
np.random.seed(1)

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class A2C(Agent):

    """ Reference: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py """

    def __init__(self, env: "AirHockey", train: bool):
        super().__init__(env)

        logger.info(f"Strategy defined for {self.name}: {self.__repr__()}")

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 4
        self.value_size = 1
        self.batch_size = config["params"]["batch_size"]

        self.frame_per_action = config["params"]["frame_per_action"]
        self.timestep_per_train = config["params"]["timestep_per_train"]
        self.iterations_on_save = config["params"]["iterations_on_save"]

        # These are hyper parameters for the Policy Gradient
        self.gamma = config["params"]["gamma"]
        self.actor_lr = config["params"]["actor_learning_rate"]
        self.critic_lr = config["params"]["critic_learning_rate"]

        # Model load and save paths
        self.actor_load_path = None if not config["actor"]["load"] else config["actor"]["load"]
        self.critic_load_path = None if not config["critic"]["load"] else config["critic"]["load"]

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        # Training epochs
        self.epochs = config["params"]["epochs"]

        # Model construction
        self.build_model()

        # Are we training?
        self.train = train

        # Keep up with the iterations
        self.t = 0

        # Exploration Strategy
        self.exploration_strategy = SoftmaxPolicy(action_size=self.action_size)

    def __repr__(self):
        return "A2C"

    def build_model(self) -> None:
        """ Create our Actor/Critic Models """

        self.actor_model, self.critic_model = model.create(
            state_size=self.state_size,
            action_size=self.action_size,
            value_size=self.value_size,
            actor_learning_rate=self.actor_lr,
            critic_learning_rate=self.critic_lr,
        )

        if self.actor_load_path and self.critic_load_path:
            self.load_model()

        logger.info("Actor Model")
        print(self.actor_model.summary())
        logger.info("Critic Model")
        print(self.critic_model.summary())
        return None

    def _get_action(self, state: "State") -> int:
        """ Using the output of policy network, pick action stochastically (Boltzmann Policy) """

        policy = self.actor_model.predict(serialize_state(state))[0]
        assert policy.shape == (self.action_size,), f"Q-values with shape {policy.shape} have the wrong dimensions"
        return self.exploration_strategy.step(policy) if self.train else np.argmax(policy)

    def discount_rewards(self, rewards: List[int]):
        """ Instead agent uses sample returns for evaluating policy
            Use TD(1) i.e. Monte Carlo updates
        """

        discounted_r, cumul_r = np.zeros_like(rewards), 0
        for t in reversed(range(0, len(rewards))):
            cumul_r = rewards[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def update(self, data: "Observation") -> None:
        """ Update policy network every episode """
        self.memory.append(data)

        # Update model in intervals
        if self.t > 0 and self.t % self.timestep_per_train == 0:

            logger.info("Update models")

            observations = self.memory.retreive()
            states = np.array([serialize_state(observation.state, dim=1) for observation in observations])
            actions = np.array([observation.action for observation in observations])
            rewards = np.array([observation.reward for observation in observations])
            episode_length = len(self.memory)
            discounted_rewards = self.discount_rewards(rewards)

            # Prediction of state values for each state appears in the episode
            values = np.array(self.critic_model.predict(states))

            # Similar to one-hot target but the "1" is replaced by Advantage Function
            # (i.e. discounted_rewards R_t - value)
            advantages = np.zeros((episode_length, self.action_size))

            for i in range(episode_length):
                advantages[i][actions[i]] = discounted_rewards[i] - values[i]
            # Train models
            self.actor_model.fit(states, advantages, epochs=self.epochs, verbose=0)
            self.critic_model.fit(states, discounted_rewards, epochs=self.epochs, verbose=0)

            # Empty buffer (treat as a cache for the minibatch)
            self.memory.purge()

        # Save model
        if self.train and self.t % self.timestep_per_train == 0:
            self.save_model()

        self.t += 1

        return None

    def load_model(self) -> None:
        """ Load a model's weights"""

        logger.info(f"Loading model from: {self.actor_load_path}")
        self.actor_model.load_weights(self.actor_load_path)

        logger.info(f"Loading model from: {self.critic_load_path}")
        self.critic_model.load_weights(self.critic_load_path)

    def save_model(self) -> None:
        """ Save a model's weights """
        logger.info(f"Saving model to: {self.path}")

        # Save actor model
        actor_path = os.path.join(self.path, "actor.h5")
        self.actor_model.save_weights(actor_path, overwrite=True)

        # Save critic model
        critic_path = os.path.join(self.path, "critic.h5")
        self.critic_model.save_weights(critic_path, overwrite=True)

""" Actor/Critic Alternate (Synchronous) """
import imp
import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from environment import AirHockey
from lib.agents import Agent
from lib.buffer import MemoryBuffer
from lib.exploration import SoftmaxPolicy
from lib.types import Observation, State
from lib.utils.helpers import serialize_state

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class A2C_1(Agent):

    """ Reference: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py """

    def __init__(self, env: "AirHockey", train: bool):
        super().__init__(env, train)

        logger.info(f"Strategy defined for {self.name}: {self.__repr__()}")

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 4
        self.value_size = 1

        # Load raw model
        path, to_load = self.model_path("a2c_1")
        model = imp.load_source("a2c_1", os.path.join(path, "model.py"))

        # Load configs
        config = model.config()
        self.batch_size = config["params"]["batch_size"]

        self.frame_per_action = config["params"]["frame_per_action"]
        self.timestep_per_train = config["params"]["timestep_per_train"]
        self.iterations_on_save = config["params"]["iterations_on_save"]

        # These are hyper parameters for the Policy Gradient
        self.gamma = config["params"]["gamma"]
        self.actor_lr = config["params"]["actor_learning_rate"]
        self.critic_lr = config["params"]["critic_learning_rate"]

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        # Training epochs
        self.epochs = config["params"]["epochs"]

        # Build models
        self.actor_model, self.critic_model = model.create(
            state_size=self.state_size,
            action_size=self.action_size,
            value_size=self.value_size,
            actor_learning_rate=self.actor_lr,
            critic_learning_rate=self.critic_lr,
        )

        if to_load:
            try:
                logger.info(f"Loading model's weights from: {path}...")
                self.actor_model.load_weights(os.path.join(path, "actor.h5"))
                self.critic_model.load_weights(os.path.join(path, "critic.h5"))
            except OSError:
                logger.info("Weights file corrupted, starting fresh...")
                pass  # If file is corrupted, move on.

        # Keep up with the iterations
        self.t = 0
        self.init = True

        # Exploration Strategy
        self.exploration_strategy = SoftmaxPolicy(action_size=self.action_size)

    def __repr__(self):
        return "A2C #2"

    def _get_action(self, state: "State") -> int:
        """ Using the output of policy network, pick action stochastically (Boltzmann Policy) """
        # Don't let LSTM spazz about not being built on first pass
        if self.train and self.init:
            return np.random.randint(0, self.action_size)

        policy = self.actor_model.predict(serialize_state(state))[0]
        assert policy.shape == (self.action_size,), f"The policy with shape {policy.shape} have the wrong dimensions"
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
        if self.train and self.t > 0 and self.t % self.timestep_per_train == 0:
            self.init = False
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
            self.actor_model.fit(
                states,
                advantages,
                epochs=self.epochs,
                verbose=0,
            )
            self.critic_model.fit(states, discounted_rewards, epochs=self.epochs, verbose=0)

            # Empty buffer (treat as a cache for the minibatch)
            self.memory.purge()

        # Save model
        if self.train and self.t % self.timestep_per_train == 0:
            self.save()

        self.t += 1

        return None

    def save(self) -> None:
        """ Save a model's weights """
        logger.info(f"Saving models to: {self.path}")

        # Save actor model
        actor_path = os.path.join(self.path, "actor.h5")
        self.actor_model.save_weights(actor_path, overwrite=True)

        # Save critic model
        critic_path = os.path.join(self.path, "critic.h5")
        self.critic_model.save_weights(critic_path, overwrite=True)

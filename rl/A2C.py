""" C51 DDQN """
import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from keras.models import load_model

from environment import AirHockey
from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.MemoryBuffer import MemoryBuffer
from rl.Networks import Networks
from utils import Observation, State, get_model_path

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class A2C(Agent):

    """ Reference: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py """

    def __init__(
        self,
        env: AirHockey,
        capacity: int,
        train: bool,
        config: Dict[str, Any]
    ):
        super().__init__(env)

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (3, int(capacity), 2)
        self.action_size = 4
        self.value_size = 1

        self.observe = config["params"]["observe"]
        self.frame_per_action = config["params"]["frame_per_action"]
        self.timestep_per_train = config["params"]["timestep_per_train"]

        # These are hyper parameters for the Policy Gradient
        self.gamma = config["params"]["gamma"]
        self.actor_lr = config["params"]["actor_learning_rate"]
        self.critic_lr = config["params"]["critic_learning_rate"]

        # Model load and save paths
        self.actor_load_path = config["actor"]["load"]
        self.actor_save_path = config["actor"]["save"]
        self.critic_load_path = config["critic"]["load"]
        self.critic_save_path = config["critic"]["save"]

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        # Model construction
        self.build_model()

        self.train = train

        # Keep up with the iterations
        self.t = 0

    def build_model(self) -> None:
        """ Create our Actor/Critic Models """

        self.actor_model = Networks().actor(
            self.state_size, self.action_size, self.actor_lr
        )
        self.critic_model = Networks().critic(
            self.state_size, self.value_size, self.critic_lr
        )

        if self.actor_load_path and self.critic_load_path:
            self.load_model()

        print("Actor Model")
        print(self.actor_model.summary())
        print("Critic Model")
        print(self.critic_model.summary())
        return None

    def get_action(self, state: State) -> int:
        """ Using the output of policy network, pick action stochastically (Stochastic Policy) """
        policy = self.actor_model.predict(np.array([state]))[0]
        if not self.train:
            return np.argmax(policy)
        return np.random.choice(self.action_size, 1, p=policy)[0]

    def discount_rewards(self, rewards: List[int]):
        """ Instead agent uses sample returns for evaluating policy
            Use TD(1) i.e. Monte Carlo updates
        """

        discounted_rewards = np.zeros_like(rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add

        return discounted_rewards

    def update(self, data: Observation) -> None:
        """ Update policy network every episode """
        self.memory.append(data)

        # Update model in intervals
        if self.t > self.observe and self.t % self.timestep_per_train == 0:
            observations = self.memory.retreive()
            states = [observation.state for observation in observations]
            actions = [observation.action for observation in observations]
            rewards = [observation.reward for observation in observations]
            episode_length = len(self.memory)

            discounted_rewards = self.discount_rewards(rewards)

            # Standardized discounted rewards
            if np.std(discounted_rewards):
                discounted_rewards = np.divide(
                    (discounted_rewards - np.mean(discounted_rewards)),
                    np.std(discounted_rewards),
                )
            else:
                self.memory.purge()
                logger.debug("Standard Deviation is Zero")
                return None

            update_inputs = np.zeros(((episode_length,) + self.state_size))

            # Episode length is like the minibatch size in DQN
            for i in range(episode_length):
                update_inputs[i, :, :, :] = states[i]

            # Prediction of state values for each state appears in the episode
            values = self.critic_model.predict(update_inputs)
            # Similar to one-hot target but the "1" is replaced by Advantage Function i.e. discounted_rewards R_t - Value
            advantages = np.zeros((episode_length, self.action_size))

            for i in range(episode_length):
                advantages[i][actions[i]] = discounted_rewards[i] - values[i]

            self.actor_model.fit(update_inputs, advantages, epochs=1, verbose=0)
            self.critic_model.fit(
                update_inputs, discounted_rewards, epochs=1, verbose=0
            )

            self.memory.purge()

        self.t += 1

        return None

    def load_model(self) -> None:
        """ Load a model"""

        logger.info(f"Loading model from: {self.actor_load_path}")

        self.actor_model = load_model(
            self.actor_load_path, custom_objects={"huber_loss": huber_loss}
        )

        logger.info(f"Loading model from: {self.critic_load_path}")

        self.critic_model = load_model(
            self.critic_load_path, custom_objects={"huber_loss": huber_loss}
        )

    def save_model(self) -> None:
        """ Save models """

        logger.info(f"Saving Actor Model to: {self.actor_save_path}")
        # Create path with epoch number
        path = get_model_path(self.actor_save_path)
        self.actor_model.save(path, overwrite=True)

        logger.info(f"Saving Critic Model to: {self.critic_save_path}")
        # Create path with epoch number
        path = get_model_path(self.critic_save_path)
        self.critic_model.save(path, overwrite=True)

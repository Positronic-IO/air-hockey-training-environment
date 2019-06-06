""" Actor/Critic (Synchronous) """
import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from keras.models import load_model

from environment import AirHockey

from rl import networks
from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.MemoryBuffer import MemoryBuffer
from rl.utils import Observation, State

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class A2C(Agent):

    """ Reference: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c.py """

    def __init__(self, env: AirHockey, capacity: int, train: bool, config: Dict[str, Any]):
        super().__init__(env)

        self.version = "0.1.0"
        logger.info(f"Strategy defined for {self.agent_name}: {self.__repr__()}")

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 4
        self.value_size = 1

        self.frame_per_action = config["params"]["frame_per_action"]
        self.timestep_per_train = config["params"]["timestep_per_train"]
        self.iterations_on_save = config["params"]["iterations_on_save"]

        # These are hyper parameters for the Policy Gradient
        self.gamma = config["params"]["gamma"]
        self.actor_lr = config["params"]["actor_learning_rate"]
        self.critic_lr = config["params"]["critic_learning_rate"]

        # Model load and save paths
        self.actor_load_path = None if not config["actor"].get("load") else config["actor"]["load"]
        self.critic_load_path = None if not config["critic"].get("load") else config["critic"]["load"]
        self.save_path = None

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        # Model construction
        self.build_model()

        self.train = train

        # Parameter Noise
        self.param_noise = True

        # Keep up with the iterations
        self.t = 0

    def build_model(self) -> None:
        """ Create our Actor/Critic Models """

        self.actor_model, self.critic_model = networks.a2c(
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

    def transfer_weights(self) -> None:
        """ Transfer model weights to target model with a factor of Tau """

        if self.param_noise:
            tau = np.random.uniform(-0.15, 0.15)
            W_actor, target_W_actor = self.actor_model.get_weights(), self.actor_model.get_weights()
            W_critic, target_W_critic = self.critic_model.get_weights(), self.critic_model.get_weights()

            for i in range(len(W_actor)):
                target_W_actor[i] = tau * W_actor[i] + (1 - tau) * target_W_actor[i]

            for i in range(len(W_actor)):
                target_W_critic[i] = tau * W_critic[i] + (1 - tau) * target_W_critic[i]

            self.actor_model.set_weights(target_W_actor)
            self.critic_model.set_weights(target_W_critic)

    def get_action(self, state: State) -> int:
        """ Using the output of policy network, pick action stochastically (Boltzmann Policy) """
        flattened_state = np.hstack(state)
        policy = self.actor_model.predict(np.array([flattened_state]))[0]
        assert policy.shape == (self.action_size,)

        if not self.train:
            return np.argmax(policy)  # Greedy policy
        return np.random.choice(np.arange(self.action_size), 1, p=policy)[0]

    def discount_rewards(self, rewards: List[int]):
        """ Instead agent uses sample returns for evaluating policy
            Use TD(1) i.e. Monte Carlo updates
        """

        discounted_r, cumul_r = np.zeros_like(rewards), 0
        for t in reversed(range(0, len(rewards))):
            cumul_r = rewards[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def update(self, data: Observation) -> None:
        """ Update policy network every episode """
        self.memory.append(data)

        # Update model in intervals
        if self.t > 0 and self.t % self.timestep_per_train == 0:

            logger.info("Update models")

            observations = self.memory.retreive()
            states = np.array([np.hstack(observation.state) for observation in observations])
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
            self.actor_model.fit(states, advantages, epochs=1, verbose=0)
            self.critic_model.fit(states, discounted_rewards, epochs=1, verbose=0)

            # Update target model
            self.transfer_weights()

            # Empty buffer (treat as a cache for the minibatch)
            self.memory.purge()

        # Save model
        if self.t % self.iterations_on_save == 0:
            self.save_model()

        self.t += 1

        return None

    def load_model(self) -> None:
        """ Load a model"""

        logger.info(f"Loading model from: {self.actor_load_path}")
        self.actor_model = load_model(self.actor_load_path, custom_objects={"huber_loss": huber_loss})

        logger.info(f"Loading model from: {self.critic_load_path}")
        self.critic_model = load_model(self.critic_load_path, custom_objects={"huber_loss": huber_loss})

    def save_model(self) -> None:
        """ Save a models """

        # Save actor model
        actor_path = os.path.join(self.save_path, "actor.h5")
        logger.info(f"Saving actor model to: {actor_path}")
        self.actor_model.save(actor_path, overwrite=True)

        # Save critic model
        critic_path = os.path.join(self.save_path, "critic.h5")
        logger.info(f"Saving critic model to: {critic_path}")
        self.critic_model.save(critic_path, overwrite=True)

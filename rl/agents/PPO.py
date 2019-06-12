""" PPO """
import logging
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from environment import AirHockey
from rl import networks
from rl.Agent import Agent
from rl.helpers import huber_loss, proximal_policy_optimization_loss, proximal_policy_optimization_loss_continuous
from rl.MemoryBuffer import MemoryBuffer
from rl.utils import Observation, State

# Set random seeds
np.random.seed(1)

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PPO(Agent):

    """ Reference: https://github.com/LuEE-C/PPO-Keras """

    def __init__(self, env: AirHockey, train: bool, config: Dict[str, Any]):
        super().__init__(env)

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

        # Model load and save paths
        self.actor_load_path = None if not config["actor"].get("load") else config["actor"]["load"]
        self.critic_load_path = None if not config["critic"].get("load") else config["critic"]["load"]
        self.save_path = None

        # Train and Save
        self.timestep_per_train = config["params"]["timestep_per_train"]
        self.iterations_on_save = config["params"]["iterations_on_save"]

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)
        self.batch_size = config["params"]["batch_size"]

        # Training epochs
        self.epochs = config["params"]["epochs"]

        # Model construction
        self.build_model()

        # Are we training?
        self.train = train

        # Noise (Continuous)
        self.noise = config.get("noise", 1)

        # Keep up with the iterations
        self.t = 0

        # Initialize
        self.action_matrix, self.policy = None, None

        self.version = "0.1.0"
        logger.info(f"Strategy defined for {self.agent_name}: {self.__repr__()}")

    def __str__(self):
        return f"{self.__class__.__name__} Continuous" if self.continuous else self.__class__.__name__

    def __repr__(self):
        return f"{self.__str__()} {self.version}"

    def build_model(self) -> None:
        """ Create our Actor/Critic Models """

        self.actor_model, self.critic_model = networks.ppo(
            state_size=self.state_size,
            action_size=self.action_size,
            actor_learning_rate=self.actor_learning_rate,
            critic_learning_rate=self.critic_learning_rate,
            continuous=self.continuous,
        )

        if self.actor_load_path and self.critic_load_path:
            self.load_model()

        logger.info("Actor Model")
        print(self.actor_model.summary())
        logger.info("Critic Model")
        print(self.critic_model.summary())
        return None

    def get_action(self, state: State) -> Union[int, Tuple[int, int]]:
        """ Return a action """

        if not self.continuous:
            return self._get_action_discrete(state)

        return self._get_action_continuous(state)

    def _get_action_discrete(self, state: State) -> int:
        """ Return a random action (discrete space) """
        policy = self.actor_model.predict(
            [np.expand_dims(np.hstack(state), axis=0), np.zeros(shape=(1, 1)), np.zeros(shape=(1, self.action_size))]
        )[0]
        # Sanity check
        assert policy.shape == (
            self.action_size,
        ), f"Policy is of shape {policy.shape} instead of {(self.action_size,)}"
        
        if self.train:
            action = np.random.choice(self.action_size, p=np.nan_to_num(policy))  # Boltzmann Policy
        else:
            action = np.argmax(policy)
        action_matrix = np.zeros(self.action_size)
        action_matrix[action] = 1
        self.action_matrix, self.policy = action_matrix, policy
        return action

    def _get_action_continuous(self, state: State) -> Tuple[int, int]:
        """ Return a random action (continuous space) """
        policy = self.actor_model.predict(
            [np.expand_dims(np.hstack(state), axis=0), np.zeros(shape=(1, 1)), np.zeros(shape=(1, self.action_size))]
        )[0]
        if self.train:
            noise = np.random.normal(0, self.noise, size=policy.shape)
            action = action_matrix = policy + noise
        else:
            action = action_matrix = policy
        self.action_matrix, self.policy = action_matrix, policy
        return action.tolist()[0]

    def discount_rewards(self, rewards: List[float]):
        """ Compute discount rewards """
        for j in range(len(rewards) - 2, -1, -1):
            rewards[j] += rewards[j + 1] * self.gamma
        return rewards

    def update(self, data: Observation) -> None:
        """ Update policy network every episode """
        self.memory.append((data, (self.action_matrix, self.policy)))

        # Update model in intervals
        if self.t > 0 and self.t % self.timestep_per_train == 0:

            logger.info("Update models")

            observations = self.memory.retreive()
            states = np.array([np.hstack(observation[0].state) for observation in observations])
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
        if self.t % self.iterations_on_save == 0:
            self.save_model()

        self.t += 1

        return None

    def load_model(self) -> None:
        """ Load a model"""

        logger.info(f"Loading model from: {self.actor_load_path}")
        self.actor_model.load_weights(self.actor_load_path)

        logger.info(f"Loading model from: {self.critic_load_path}")
        self.critic_model.load_weights(self.critic_load_path)

    def save_model(self) -> None:
        """ Save a models """

        path = "_continuous" if self.continuous else ""

        # Save actor model
        actor_path = os.path.join(self.save_path, f"actor{path}.h5")
        logger.info(f"Saving actor weights to: {actor_path}")
        self.actor_model.save_weights(actor_path, overwrite=True)

        # Save critic model
        critic_path = os.path.join(self.save_path, f"critic{path}.h5")
        logger.info(f"Saving critic weights to: {critic_path}")
        self.critic_model.save_weights(critic_path, overwrite=True)

""" C51 DDQN """
import imp
import logging
import math
import os
from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from lib.agents import Agent
from lib.buffer import MemoryBuffer
from lib.exploration import EpsilonGreedy
from lib.types import Observation, State
from lib.utils.helpers import serialize_state

# Set random seeds
np.random.seed(1)

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class c51(Agent):

    """ Reference: https://github.com/flyyufelix/C51-DDQN-Keras """

    def __init__(self, env: "AirHockey", train: bool):
        super().__init__(env)

        logger.info(f"Strategy defined for {self.name}: {self.__repr__()}")

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 4

        # Load raw model
        path, to_load = self.model_path("c51")
        model = imp.load_source("c51", os.path.join(path, "model.py"))

        # Load configs
        self.config = model.config()

        # These are the hyper parameters for the c51
        self.gamma = self.config["params"]["gamma"]
        self.learning_rate = self.config["params"]["learning_rate"]
        self.batch_size = self.config["params"]["batch_size"]
        self.frame_per_action = self.config["params"]["frame_per_action"]
        self.update_target_freq = self.config["params"]["update_target_freq"]
        self.timestep_per_train = self.config["params"]["timestep_per_train"]
        self.timestep_per_train = self.config["params"]["timestep_per_train"]

        # Initialize Atoms
        self.num_atoms = self.config["params"]["num_atoms"]  # Defaults to 51 for C51
        self.v_max = self.config["params"]["v_max"]  # Max possible score for agents is 10
        self.v_min = self.config["params"]["v_min"]
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Build models
        self.model = model.create(self.state_size, self.action_size, self.num_atoms, self.learning_rate)
        self.target_model = model.create(self.state_size, self.action_size, self.num_atoms, self.learning_rate)

        # Parameter Noise
        self.param_noise = True

        if to_load:
            try:
                logger.info(f"Loading model's weights from: {path}...")
                self.model.load_weights(os.path.join(path, "model.h5"))
            except OSError:
                logger.info("Weights file corrupted, starting fresh...")
                pass  # If file is corrupted, move on.

        # Transfer weights
        self.transfer_weights()
        logger.info(self.model.summary())

        # create replay memory using deque
        self.max_memory = self.config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        # We want to choose our prediction more than a random policy.
        self.train = train

        # Training epochs
        self.epochs = self.config["params"]["epochs"]

        # Keep up with the iterations
        self.t = 0

        # Exploration strategy
        self.exploration_strategy = EpsilonGreedy(action_size=self.action_size)

    def __repr__(self) -> str:
        return "c51 DDQN"

    def transfer_weights(self) -> None:
        """ Transfer model weights to target model with a factor of Tau """

        if self.param_noise:
            tau = np.random.uniform(0, 0.2)
            W, target_W = self.model.get_weights(), self.target_model.get_weights()
            for i in range(len(W)):
                target_W[i] = tau * W[i] + (1 - tau) * target_W[i]
            self.target_model.set_weights(target_W)
            return None

        self.target_model.set_weights(self.model.get_weights())
        return None

    def update_target_model(self) -> None:
        """ After some time interval update the target model to be same with model """

        # Update the target model to be same with model
        if self.t > 0 and self.t % self.update_target_freq == 0:
            # Transfer weights
            logger.info("Sync target model for c51")
            self.transfer_weights()
        return None

    def _get_action(self, state: "State") -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Compute rewards for any posible action
        z = self.model.predict(serialize_state(state))
        z_concat = np.vstack(z)
        q_values = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        assert q_values.shape == (self.action_size,), f"Q-values with shape {q_values.shape} have the wrong dimensions"

        return self.exploration_strategy.step(q_values) if self.train else np.argmax(q_values)

    def update(self, data: "Observation") -> None:
        """ Experience replay """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Sync Target Model
        self.update_target_model()

        # Update model in intervals
        if self.t > 0 and self.t % self.timestep_per_train == 0:

            logger.info(f"Updating c51 model")

            # Get samples from replay
            num_samples = min(self.batch_size, len(self.memory))
            replay_samples = self.memory.sample(num_samples)

            # Convert Observations/trajectories into tensors
            action = np.array([sample[1] for sample in replay_samples], dtype=np.int32)
            reward = np.array([sample[2] for sample in replay_samples], dtype=np.float64)
            done = np.array([1 if sample[3] else 0 for sample in replay_samples], dtype=np.int8)

            state_inputs = np.array([serialize_state(sample[0], dim=1) for sample in replay_samples])
            next_states = np.array([serialize_state(sample[4], dim=1) for sample in replay_samples])

            assert state_inputs.shape == (
                (num_samples,) + self.state_size
            ), f"state_inputs shape is {state_inputs.shape} when it was sipposed to be {((num_samples,) + self.state_size)}"
            assert next_states.shape == (
                (num_samples,) + self.state_size
            ), f"next_states shape is {next_states.shape} when it was sipposed to be {((num_samples,) + self.state_size)}"

            # Initiate q-value distribution
            m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(self.action_size)]

            z = self.model.predict(next_states)
            z_ = self.target_model.predict(next_states)

            # Get Optimal Actions for the next states (from distribution z)
            z_concat = np.vstack(z)
            q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)  # length (num_atoms x num_actions)
            q = q.reshape((num_samples, self.action_size), order="F")
            optimal_action_idxs = np.argmax(q, axis=1)

            # Project Next State Value Distribution (of optimal action) to Current State
            for i in range(num_samples):
                if done[i]:  # Terminal State
                    # Distribution collapses to a single point
                    Tz = min(self.v_max, max(self.v_min, reward[i]))
                    bj = (Tz - self.v_min) / self.delta_z
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[action[i]][i][int(m_l)] += m_u - bj
                    m_prob[action[i]][i][int(m_u)] += bj - m_l
                else:
                    for j in range(self.num_atoms):
                        Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
                        bj = (Tz - self.v_min) / self.delta_z
                        m_l, m_u = math.floor(bj), math.ceil(bj)
                        m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
                        m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)

            self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

        # Save model
        if self.train and self.t % self.timestep_per_train == 0:
            self.save()

        self.t += 1
        return None

    def save(self) -> None:
        """ Save a model's weights """
        logger.info(f"Saving model to: {self.path}")
        path = os.path.join(self.path, "model.h5")
        self.model.save_weights(path, overwrite=True)

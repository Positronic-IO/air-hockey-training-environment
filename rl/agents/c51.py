""" C51 DDQN """
import logging
import math
import os
from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from rl import networks
from rl.Agent import Agent
from rl.MemoryBuffer import MemoryBuffer
from rl.utils import Observation, State

# Set random seeds
np.random.seed(1)

# Initiate Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class c51(Agent):

    """ Reference: https://github.com/flyyufelix/C51-DDQN-Keras """

    def __init__(self, env: AirHockey, train: bool, config: Dict[str, Any]):
        super().__init__(env)

        self.version = "0.2.0"
        logger.info(f"Strategy defined for {self.agent_name}: {self.__repr__()}")

        # Get size of state and action
        # State grows by the amount of frames we want to hold in our memory
        self.state_size = (1, 8)
        self.action_size = 4

        # These are the hyper parameters for the c51
        self.gamma = config["params"]["gamma"]
        self.learning_rate = config["params"]["learning_rate"]
        self.epsilon = config["params"]["epsilon"]
        self.initial_epsilon = config["params"]["initial_epsilon"]
        self.final_epsilon = config["params"]["final_epsilon"]
        self.batch_size = config["params"]["batch_size"]
        self.observe = config["params"]["observe"]
        self.explore = config["params"]["explore"]
        self.frame_per_action = config["params"]["frame_per_action"]
        self.update_target_freq = config["params"]["update_target_freq"]
        self.timestep_per_train = config["params"]["timestep_per_train"]
        self.iterations_on_save = config["params"]["iterations_on_save"]

        # Initialize Atoms
        self.num_atoms = config["params"]["num_atoms"]  # Defaults to 51 for C51
        self.v_max = config["params"]["v_max"]  # Max possible score for agents is 10
        self.v_min = config["params"]["v_min"]
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # create replay memory using deque
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        # If we are not training, set our epsilon to final_epsilon.
        # We want to choose our prediction more than a random policy.
        self.train = train
        self.epsilon = self.epsilon if self.train else self.final_epsilon

        # Keep up with the iterations
        self.t = 0

        # Model load and save paths
        self.load_path = None if not config.get("load") else config.get("load")
        self.save_path = None

        # Parameter Noise
        self.param_noise = True

        # Model construction
        self.build_model()

    def __repr__(self) -> str:
        return f"{self.__str__()} {self.version}"

    def __str__(self) -> str:
        return "c51 DDQN"

    def transfer_weights(self) -> None:
        """ Transfer model weights to target model with a factor of Tau """

        if self.param_noise:
            tau = np.random.uniform(-0.15, 0.15)
            W, target_W = self.model.get_weights(), self.target_model.get_weights()
            for i in range(len(W)):
                target_W[i] = tau * W[i] + (1 - tau) * target_W[i]
            self.target_model.set_weights(target_W)
            return None

        self.target_model.set_weights(self.model.get_weights())
        return None

    def build_model(self) -> None:
        """ Create our DNN model for Q-value approximation """

        model = networks.c51(self.state_size, self.action_size, self.learning_rate)

        self.model = model
        self.target_model = model

        if self.load_path:
            self.load_model()

        # Transfer weights
        self.transfer_weights()

        print(self.model.summary())
        return model

    def update_target_model(self) -> None:
        """ After some time interval update the target model to be same with model """

        # Update the target model to be same with model
        if self.t > 0 and self.t % self.update_target_freq == 0:
            # Transfer weights
            logger.info("Sync target model for c51")
            self.transfer_weights()
        return None

    def _epsilon(self) -> None:
        """ Update all things epsilon """

        # If we are not in training mode, then break.
        if not self.train:
            return None

        if self.epsilon > self.final_epsilon and self.t % self.observe == 0:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

        return None

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            idx = np.random.randint(0, self.action_size)
            return idx

        flattened_state = np.hstack(state)
        # Compute rewards for any posible action
        z = self.model.predict(np.array([flattened_state]), batch_size=1)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        # Pick action with the biggest Q value
        idx = np.argmax(q)
        return idx

    def update(self, data: Observation) -> None:
        """ Experience replay """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Modify epsilon
        self._epsilon()

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

            state_inputs = np.array([np.hstack(sample[0]) for sample in replay_samples])
            next_states = np.array([np.hstack(sample[4]) for sample in replay_samples])

            assert state_inputs.shape == ((num_samples,) + self.state_size)
            assert next_states.shape == ((num_samples,) + self.state_size)

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

            self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, epochs=1, verbose=0)

        # Save model
        if self.t % self.iterations_on_save == 0:
            self.save_model()

        self.t += 1

        return None

    def load_model(self) -> None:
        """ Load a model"""

        logger.info(f"Loading model's weights from: {self.load_path}")

        self.model.load_weights(self.load_path)

    def save_model(self) -> None:
        """ Save a model's weights """

        # Create path with epoch number
        path = os.path.join(self.save_path, "dueling.h5")
        logger.info(f"Saving model to: {self.save_path}")
        self.model.save_weights(path, overwrite=True)

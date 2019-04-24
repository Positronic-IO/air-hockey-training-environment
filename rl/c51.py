""" C51 DDQN """

import math
import os
import random
import time
from collections import deque
from typing import Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.Networks import Networks
from utils import Observation, State, get_model_path


class c51(Agent):

    """ Reference: https://github.com/flyyufelix/C51-DDQN-Keras """

    def __init__(
        self,
        env: AirHockey,
        config: Dict[str, Dict[str, int]],
        agent_name: str = "main",
    ):
        super().__init__(env, agent_name)

        # get size of state and action
        self.state_size = (7, 2)
        self.action_size = len(self.env.actions)

        # these is hyper parameters for the DQN
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
        self.timestep_per_train = config["params"][
            "timestep_per_train"
        ]  # Number of timesteps between training interval

        # Initialize Atoms
        self.num_atoms = config["params"]["num_atoms"]  # Defaults to51 for C51
        self.v_max = config["params"][
            "v_max"
        ]  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = config["params"]["v_min"]
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # create replay memory using deque
        self.memory = deque()
        self.max_memory = config["params"][
            "max_memory"
        ]  # number of previous transitions to remember

        # Counters
        self.batch_counter, self.sync_counter, self.t = 0, 0, 0

        # Model construction
        self.build_model()

        self.version = "0.1.0"

    def build_model(self) -> None:
        """ Create our DNN model for Q-value approximation """

        model = Networks().c51(self.state_size, self.action_size, self.learning_rate)

        self.model = model
        self.target_model = model
        print(self.model.summary())
        return model

    def update_target_model(self) -> None:
        """ Copy weights from model to target_model """

        print("Sync target model")
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)

        # Compute rewards for any posible action
        z = self.model.predict(np.array([state]), batch_size=1)
        z_concat = np.vstack(z)
        q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1)
        # Pick action with the biggest Q value
        idx = np.argmax(q[0])
        return idx

    def remember(self, data: Observation) -> None:
        """ Push data into memory for replay later """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Modify epsilon
        if self.epsilon > self.final_epsilon and self.t > self.observe:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        self.t += 1

        # Memory management
        if len(self.memory) > self.max_memory:
            self.memory.popleft()

        assert len(self.memory) < self.max_memory + 1, "Max memory exceeded"

    def update(self, data: Observation) -> Union[float, None]:
        """ Experience replay """

        # Push data into observation and remove one from buffer
        self.remember(data)

        self.sync_counter += 1
        if self.sync_counter > self.update_target_freq:
            # Sync Target Model
            self.update_target_model()
            self.sync_counter = 0

        # Update model in intervals
        self.batch_counter += 1
        if self.batch_counter > self.timestep_per_train:

            print("Update Model")

            # Reset Batch counter
            self.batch_counter = 0

            # Get samples from replay
            num_samples = min(
                self.batch_size * self.timestep_per_train, len(self.memory)
            )
            replay_samples = random.sample(self.memory, num_samples)

            # Convert Observations/trajectories into tensors
            action = np.array([sample[1] for sample in replay_samples], dtype=np.int32)
            reward = np.array(
                [sample[2] for sample in replay_samples], dtype=np.float64
            )
            done = np.array(
                [1 if sample[3] else 0 for sample in replay_samples], dtype=np.int8
            )

            state_inputs = np.array([sample[0] for sample in replay_samples])
            next_states = np.array([sample[4] for sample in replay_samples])

            assert state_inputs.shape == ((num_samples,) + self.state_size)
            assert next_states.shape == ((num_samples,) + self.state_size)

            # Initiate q-value distribution
            m_prob = [
                np.zeros((num_samples, self.num_atoms)) for i in range(self.action_size)
            ]

            z = self.model.predict(next_states)
            z_ = self.target_model.predict(next_states)

            # Get Optimal Actions for the next states (from distribution z)
            z_concat = np.vstack(z)
            q = np.sum(
                np.multiply(z_concat, np.array(self.z)), axis=1
            )  # length (num_atoms x num_actions)
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
                        Tz = min(
                            self.v_max,
                            max(self.v_min, reward[i] + self.gamma * self.z[j]),
                        )
                        bj = (Tz - self.v_min) / self.delta_z
                        m_l, m_u = math.floor(bj), math.ceil(bj)
                        m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][
                            j
                        ] * (m_u - bj)
                        m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][
                            j
                        ] * (bj - m_l)

            loss = self.model.fit(
                state_inputs, m_prob, batch_size=self.batch_size, epochs=1, verbose=0
            )

            return loss.history["loss"]
        return None

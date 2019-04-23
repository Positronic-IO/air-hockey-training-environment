""" C51 DDQN Algorithm"""

import math
import os
import random
import time
from collections import deque
from typing import Tuple, Union

import numpy as np

from rl.Agent import Agent
from rl.helpers import huber_loss
from rl.Networks import Networks
from utils import Observation, State, get_model_path


class c51(Agent):

    """ Reference: https://github.com/flyyufelix/C51-DDQN-Keras """

    def __init__(self, env, agent_name="main"):
        super().__init__(env, agent_name)

        # get size of state and action
        self.state_size = (7, 2)
        self.action_size = len(self.env.actions)

        # these is hyper parameters for the DQN
        self.gamma = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1.0
        self.initial_epsilon = 1.0
        self.final_epsilon = 1
        self.batch_size = 32
        self.observe = 2000
        self.explore = 50000
        self.frame_per_action = 4
        self.update_target_freq = 3000
        self.timestep_per_train = 10000  # Number of timesteps between training interval

        # Initialize Atoms
        self.num_atoms = 51  # 51 for C51
        self.v_max = (
            10
        )  # Max possible score for Defend the center is 26 - 0.1*26 = 23.4
        self.v_min = -10  # -0.1*26 - 1 = -3.6
        self.delta_z = (self.v_max - self.v_min) / float(self.num_atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

        # Create replay memory using deque
        self.memory = deque()
        self.max_memory = 50000  # number of previous transitions to remember

        # Counters
        self.batch_counter = 0
        self.sync_counter = 0
        self.t = 0

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

            num_samples = min(
                self.batch_size * self.timestep_per_train, len(self.memory)
            )
            replay_samples = random.sample(self.memory, num_samples)

            state_inputs = np.zeros(((num_samples,) + self.state_size))
            next_states = np.zeros(((num_samples,) + self.state_size))
            m_prob = [
                np.zeros((num_samples, self.num_atoms)) for i in range(self.action_size)
            ]
            action, reward, done = list(), list(), list()
            for i in range(num_samples):
                state_inputs[i, :, :] = replay_samples[i][0]
                action.append(replay_samples[i][1])
                reward.append(replay_samples[i][2])
                done.append(replay_samples[i][3])
                next_states[i, :, :] = replay_samples[i][4]

            z = self.model.predict(next_states)
            z_ = self.target_model.predict(next_states)

            # Get Optimal Actions for the next states (from distribution z)
            optimal_action_idxs = list()
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

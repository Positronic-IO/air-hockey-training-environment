""" Dueling DDQN """

import os
import random
import time
from collections import deque
from typing import Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from rl.Agent import Agent
from rl.helpers import TensorBoardLogger, huber_loss
from rl.MemoryBuffer import MemoryBuffer
from rl.Networks import Networks
from utils import Observation, State, get_model_path


class DuelingDDQN(Agent):

    """ Reference: https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/dueling_ddqn.py """

    def __init__(
        self,
        env: AirHockey,
        config: Dict[str, Dict[str, int]],
        tbl: TensorBoardLogger,
        agent_name: str = "main",
    ):
        super().__init__(env, agent_name)

        # get size of state and action
        self.state_size = (3, 4, 2)
        self.action_size = len(self.env.actions)

        # these is hyper parameters for the Double DQN
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

        # Initialize replay buffer
        self.max_memory = config["params"]["max_memory"]
        self.memory = MemoryBuffer(self.max_memory)

        # Model construction
        self.build_model()

        # Counters
        self.batch_counter, self.sync_counter, self.t = 0, 0, 0

        # Initiate Tensorboard
        self.tbl = tbl

        self.version = "0.1.0"

    def build_model(self) -> None:
        """ Create our DNN model for Q-value approximation """

        model = Networks().dueling_ddqn(
            self.state_size, self.action_size, self.learning_rate
        )

        self.model = model
        self.target_model = model
        print(self.model.summary())
        return None

    def update_target_model(self) -> None:
        """ After some time interval update the target model to be same with model """

        # Update the target model to be same with model
        self.sync_counter += 1
        if self.sync_counter > self.update_target_freq:
            # Sync Target Model
            self.sync_counter = 0

            print("Sync target model")
            self.target_model.set_weights(self.model.get_weights())

        return None

    def _epsilon(self) -> None:
        """ Update all things epsilon """

        self.tbl.log_scalar(
            f"{self.__class__.__name__.title()} epsilon", self.epsilon, self.t
        )

        if self.epsilon > self.final_epsilon and self.t % self.observe == 0:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore
        self.t += 1

    def get_action(self, state: State) -> int:
        """ Apply an espilon-greedy policy to pick next action """

        # Helps over fitting, encourages to exploration
        if np.random.uniform(0, 1) < self.epsilon:
            idx = np.random.randint(0, self.action_size)
            self.tbl.log_histogram(
                f"{self.__class__.__name__.title()} Greedy Actions", idx, self.t
            )
            return idx

        # Compute rewards for any posible action
        rewards = self.model.predict(np.array([state]), batch_size=1)[0]
        assert len(rewards) == self.action_size

        idx = np.argmax(rewards)
        self.tbl.log_histogram(
            f"{self.__class__.__name__.title()} Predict Actions", idx, self.t
        )
        return idx

    def update(self, data: Observation) -> None:
        """ Experience replay """

        # Push data into observation and remove one from buffer
        self.memory.append(data)

        # Update the target model to be same with model
        self.update_target_model()

        # Update model in intervals
        self.batch_counter += 1
        if self.batch_counter > self.timestep_per_train:
            self.batch_counter = 0

            print("Update Model")

            # Get samples from replay
            num_samples = min(
                self.batch_size * self.timestep_per_train, len(self.memory)
            )
            replay_samples = self.memory.sample(num_samples)

            # Convert Observations/trajectories into tensors
            action = np.array([sample[1] for sample in replay_samples], dtype=np.int32)
            reward = np.array(
                [sample[2] for sample in replay_samples], dtype=np.float64
            )
            done = np.array(
                [1 if sample[3] else 0 for sample in replay_samples], dtype=np.int8
            )

            update_input = np.array([sample[0] for sample in replay_samples])
            update_target = np.array([sample[4] for sample in replay_samples])

            assert update_input.shape == ((num_samples,) + self.state_size)
            assert update_target.shape == ((num_samples,) + self.state_size)

            target = self.model.predict(update_input)
            target_val = self.model.predict(update_target)
            target_val_ = self.target_model.predict(update_target)

            for i in range(num_samples):
                # like Q Learning, get maximum Q value at s'
                # But from target model

                old_t = target[i][action[i]]

                if done[i]:
                    target[i][action[i]] = reward[i]
                else:
                    # the key point of Double DQN
                    # selection of action is from model
                    # update is from target model
                    a = np.argmax(target_val[i])
                    target[i][action[i]] = reward[i] + self.gamma * (target_val_[i][a])

            loss = self.model.fit(
                update_input, target, batch_size=self.batch_size, epochs=1, verbose=0
            )

        # Modify epsilon
        self._epsilon()

        return None

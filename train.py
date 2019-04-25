""" Air Hockey Simulator """
import json
import os
import random
import sys
import time
from typing import Dict, Union

import numpy as np

from environment import AirHockey
from rl import Strategy, MemoryBuffer
from utils import Observation, State, get_config, get_model_path, write_results

try:
    import sentry_sdk

    # sentry_sdk.init(os.getenv("sentry"))
except ImportError:
    pass


class Train:
    def __init__(self):

        # Load main config
        self.config = get_config()

        # Set up game environment
        self.env = AirHockey()

        # Set up our agent
        self.agent = Strategy().make(
            self.config["training"]["agent"]["strategy"], self.env
        )
        self._agent_load_save()

        # Set up opponent (if not the "basic" ai)
        if self.config["training"]["opponent"]["strategy"] != "basic":
            self.opponent = Strategy().make(
                self.config["training"]["opponent"]["strategy"],
                self.env,
                agent_name="opponent",
            )

            self._opponent_load_save()

        # Interesting and important constants
        self.epoch = 0
        self.iterations_on_save = 10 ** 4
        self.iterations = 1
        self.new = False

        # We begin..
        self.init, self.init_opponent = True, True

        # Cumulative scores
        self.agent_cumulative_score, self.opponent_cumulative_score = 0, 0

        # Set up buffers for agent position, puck position, opponent position
        self.agent_location_buffer = MemoryBuffer(self.config["capacity"], [0,0])
        self.puck_location_buffer = MemoryBuffer(self.config["capacity"], [0,0])
        self.opponent_location_buffer = MemoryBuffer(self.config["capacity"], [0,0])

        # Update buffers
        self._update_buffers()

    def _agent_load_save(self) -> None:
        """ Load/Save models for agent """

        # If we pass a weights file, load it.
        if (
            hasattr(self.agent, "load_model")
            and self.config["training"]["agent"]["load"]
        ):
            self.agent.load_path = get_model_path(
                self.config["training"]["agent"]["load"]
            )
            self.agent.load_model()

        if (
            hasattr(self.agent, "save_model")
            and not self.config["training"]["agent"]["save"]
            and not self.config["training"]["agent"]["load"]
        ):
            print("Please specify a path to save model.")
            sys.exit()

        return None

    def _opponent_load_save(self) -> None:
        """ Load/Save models for opponent """

        # If we pass a weights file, load it.
        if (
            hasattr(self.opponent, "load_model")
            and self.config["training"]["opponent"]["load"]
        ):
            self.opponent.load_path = get_model_path(
                self.config["training"]["opponent"]["load"]
            )
            self.opponent.load_model()

        if (
            hasattr(self.opponent, "save_model")
            and not self.config["training"]["opponent"]["save"]
            and not self.config["training"]["opponent"]["load"]
        ):
            print("Please specify a path to save model.")
            sys.exit()

        return None

    def _update_buffers(self) -> None:
        """ Update redis buffers """

        self.puck_location = json.loads(self.env.redis.get("puck"))["location"]
        self.agent_location = json.loads(self.env.redis.get("agent_mallet"))["location"]
        self.opponent_location = json.loads(self.env.redis.get("opponent_mallet"))[
            "location"
        ]

        self.agent_location_buffer.append(self.agent_location)
        self.puck_location_buffer.append(self.puck_location)
        self.opponent_location_buffer.append(self.opponent_location)

        return None

    def stats(self) -> None:
        """ Record training stats """

        results = dict()

        if self.env.agent_cumulative_score > self.agent_cumulative_score:
            results["agent"] = [self.env.agent_cumulative_score]
            self.agent_cumulative_score = self.env.agent_cumulative_score
            self.new = True
        else:
            results["agent"] = [self.agent_cumulative_score]

        if self.env.cpu_cumulative_score > self.opponent_cumulative_score:
            results["opponent"] = [self.env.cpu_cumulative_score]
            self.opponent_cumulative_score = self.env.cpu_cumulative_score
            self.new = True
        else:
            results["opponent"] = [self.opponent_cumulative_score]

        if self.env.agent_score == 10:
            results["agent_win"] = [1]
        else:
            results["agent_win"] = [0]

        if self.env.cpu_score == 10:
            results["cpu_win"] = [1]
        else:
            results["cpu_win"] = [0]

        # results["reward_per_episode"] = [self.env.reward_per_episode]

        if self.new:
            write_results(self.config["training"]["results"], results)
            self.env.reward_per_episode = 0
            self.new = False

        return None

    def main_player(self) -> None:
        """ Main player """

        # For first move, move in a random direction
        if self.init:
            action = np.random.randint(0, len(self.env.actions) - 1)

            # Update game state
            self.agent.move(action)

            self.init = False
        else:
            # Now, let the model do all the work

            # Update buffers
            self._update_buffers()

            # Current state
            state = State(
                agent_location=self.agent_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                opponent_location=self.opponent_location_buffer.retreive(),
            )

            # Determine next action
            action = self.agent.get_action(state)

            # Update game state
            self.agent.move(action)

            # Update buffers
            self._update_buffers()

            # New state
            new_state = State(
                agent_location=self.agent_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                opponent_location=self.opponent_location_buffer.retreive(),
            )

            # Record reward
            self.env.reward_per_episode += self.env.reward

            # Observation of the game at the moment
            observation = Observation(
                state=state,
                action=action,
                reward=self.env.reward,
                done=self.env.done,
                new_state=new_state,
            )

            # Update model
            self.agent.update(observation)

            # Save results to csv
            if self.config["training"]["results"]:
                self.stats()

        # After so many iterations, save motedel
        if (
            hasattr(self.agent, "save_model")
            and self.iterations % self.iterations_on_save == 0
        ):
            if self.config["training"]["agent"]["save"]:
                self.agent.save_path = get_model_path(
                    self.config["training"]["agent"]["save"]
                )
                self.agent.save_model(self.epoch)
            else:
                self.agent.save_model(epoch=self.epoch)
            self.epoch += 1

        return None

    def opponent_player(self) -> None:
        """ Opponent player """

        # Update buffers
        self._update_buffers()

        if self.config["training"]["opponent"]["strategy"] == "basic":
            while self.env.ticks_to_ai == 0:
                self.env.mallet_ai()
                self.env.ticks_to_ai = 10
            self.env.opponent.update_mallet()
        else:

            # # For first move, move in a random direction
            if self.init_opponent:

                action = np.random.randint(0, len(self.env.actions) - 1)

                # Update game state
                self.opponent.move(action)

                self.init_opponent = False
            else:
                # Now, let the model do all the work

                # Update buffers
                self._update_buffers()

                # Current state
                state = State(
                    agent_location=self.opponent_location_buffer.retreive(),
                    puck_location=self.puck_location_buffer.retreive(),
                    opponent_location=self.agent_location_buffer.retreive(),
                )

                # Determine next action
                action = self.opponent.get_action(state)

                # Update game state
                self.opponent.move(action)

                # Update buffers
                self._update_buffers()

                # New state
                new_state = State(
                    agent_location=self.opponent_location_buffer.retreive(),
                    puck_location=self.puck_location_buffer.retreive(),
                    opponent_location=self.agent_location_buffer.retreive(),
                )

                # Observation of the game at the moment
                observation = Observation(
                    state=state,
                    action=action,
                    reward=(
                        -1 * self.env.reward
                    ),  # Opposite reward of our agent, only works for current reward settings
                    done=self.env.done,
                    new_state=new_state,
                )

                # Update model
                self.opponent.update(observation)

                # # After so many iterations, save motedel
                if (
                    hasattr(self.opponent, "save_model")
                    and self.iterations % self.iterations_on_save == 0
                ):
                    if self.config["training"]["opponent"]["save"]:
                        self.opponent.save_path = get_model_path(
                            self.config["training"]["opponent"]["save"]
                        )
                        self.opponent.save_model(self.epoch - 1)
                    else:
                        self.opponent.save_model(epoch=self.epoch - 1)

        return None

    def run(self) -> None:
        """ Main guts of training """

        # Game loop
        while True:

            # Our Agent
            self.main_player()

            # Our opponent
            self.opponent_player()

            # Update iterator
            self.iterations += 1

            # Compute scores
            if self.env.cpu_score == 10:
                print(f"Computer {self.env.cpu_score}, agent {self.env.agent_score}")
                print("Computer wins!")
                self.env.reset(total=True)

            if self.env.agent_score == 10:
                print(f"Computer {self.env.cpu_score}, agent {self.env.agent_score}")
                print("Agent wins!")
                self.env.reset(total=True)


if __name__ == "__main__":
    # Run program
    train = Train()
    train.run()

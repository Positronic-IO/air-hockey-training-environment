""" Air Hockey Simulator """
import json
import os
import random
import sys
import time
from typing import Dict, Union

import numpy as np

from environment import AirHockey
from rl import MemoryBuffer, Strategy
from utils import Observation, State, get_config, get_model_path, write_results, parse_args

# try:
#     if os.getenv("debug"):
#         import sentry_sdk
#         sentry_sdk.init(os.getenv("sentry"))
# except ImportError:
#     pass

from connect import Connection

class Train:

    conn = Connection().make("server", test=True)

    def __init__(self):
        
        # Parse cli args
        self.args = parse_args()

        # Set up our robot
        self.robot = Strategy().make(strategy=self.args["robot"], capacity=self.args["capacity"])
        self.robot.agent_name = "robot"

        self.opponent = Strategy().make(strategy=self.args["robot"], capacity=self.args["capacity"])
        self.opponent.agent_name = "opponent"

        # Interesting and important constants
        self.epoch = 0
        self.iterations_on_save = 10 ** 4
        self.iterations = 1
        self.new = False

        # We begin..
        self.init, self.init_opponent = True, True

        # Cumulative scores
        self.robot_cumulative_score, self.opponent_cumulative_score = 0, 0

        # Cumulative wins
        self.robot_cumulative_win, self.opponent_cumulative_win = 0, 0

        # Set up buffers for agent position, puck position, opponent position
        self.robot_location_buffer = MemoryBuffer(self.args["capacity"], (0, 0))
        self.puck_location_buffer = MemoryBuffer(self.args["capacity"], (0, 0))
        self.opponent_location_buffer = MemoryBuffer(self.args["capacity"], (0, 0))

        self.data = dict()
        # Update buffers
        self._update_buffers()

        # Initial time
        self.time = time.time()
        self.wait = (60 ** 2) * int(self.args["time"])  # Defaults to 3 hours

    

    def _update_buffers(self) -> None:
        """ Update redis buffers """

        self.data = self.conn.get("all")
        self.puck_location = self.data["puck"]
        self.robot_location = self.data["robot"]
        self.opponent_location = self.data["opponent"]

        self.robot_location_buffer.append(tuple(self.robot_location))
        self.puck_location_buffer.append(tuple(self.puck_location))
        self.opponent_location_buffer.append(tuple(self.opponent_location))

        return None

    # def stats(self) -> None:
    #     """ Record training stats """

    #     results = dict()

    #     if self.env.agent_cumulative_score > self.agent_cumulative_score:
    #         results["agent_goal"] = [self.env.agent_cumulative_score]
    #         self.agent_cumulative_score = self.env.agent_cumulative_score
    #         self.new = True
    #     else:
    #         results["agent_goal"] = [self.agent_cumulative_score]

    #     if self.env.cpu_cumulative_score > self.opponent_cumulative_score:
    #         results["opponent_goal"] = [self.env.cpu_cumulative_score]
    #         self.opponent_cumulative_score = self.env.cpu_cumulative_score
    #         self.new = True
    #     else:
    #         results["opponent_goal"] = [self.opponent_cumulative_score]

    #     if self.env.agent_score == 10:
    #         results["agent_win"] = [1]
    #         self.agent_cumulative_win += 1
    #     else:
    #         results["agent_win"] = [0]

    #     if self.env.cpu_score == 10:
    #         results["opponent_win"] = [1]
    #         self.opponent_cumulative_win += 1
    #     else:
    #         results["opponent_win"] = [0]

    #     if self.new:
    #         write_results(self.config["training"]["results"], results)
    #         self.new = False

    #         # Push to Tensorboard
    #         self.tbl.log_scalar(
    #             f"Agent Win", self.agent_cumulative_win, self.iterations
    #         )
    #         self.tbl.log_scalar(
    #             f"Opponent Win", self.opponent_cumulative_win, self.iterations
    #         )
    #         self.tbl.log_scalar(
    #             f"Agent goal", results["agent_goal"][0], self.iterations
    #         )
    #         self.tbl.log_scalar(
    #             f"Opponent goal", results["opponent_goal"][0], self.iterations
    #         )
    #         self.tbl.log_scalar(
    #             f"Agent Win", self.agent_cumulative_win, self.iterations
    #         )
    #         self.tbl.log_scalar(
    #             f"Agent Cumulative Reward",
    #             self.env.agent_cumulative_reward,
    #             self.iterations,
    #         )
    #         self.tbl.log_scalar(
    #             f"Opponent Cumulative Reward",
    #             self.env.cpu_cumulative_reward,
    #             self.iterations,
    #         )

    #     return None

    def main_player(self) -> None:
        """ Main player """

        # For first move, move in a random direction
        if self.init:
            action = np.random.randint(0, 3)

            # Update game state
            self.robot.move(action)

            self.init = False
        else:
            # Now, let the model do all the work

            # Update buffers
            self._update_buffers()

            # Current state
            state = State(
                robot_location=self.robot_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                opponent_location=self.opponent_location_buffer.retreive(),
            )

            # Determine next action
            action = self.robot.get_action(state)

            # Update game state
            self.robot.move(action)

            # Update buffers
            self._update_buffers()

            # New state
            new_state = State(
                robot_location=self.robot_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                opponent_location=self.opponent_location_buffer.retreive(),
            )

            # Get updated stats

            # Observation of the game at the moment
            observation = Observation(
                state=state,
                action=action,
                reward=self.data["reward"],
                done=self.data["done"],
                new_state=new_state,
            )

            # Update model
            self.robot.update(observation)

            # Save results to csv
            # if self.config["training"]["results"]:
            #     self.stats()

        # After so many iterations, save model
        if self.iterations % self.iterations_on_save == 0:
            self.robot.save_model()

        return None

    def opponent_player(self) -> None:
        """ Opponent player """

        # For first move, move in a random direction
        if self.init_opponent:

            action = np.random.randint(0, 3)

            # Update game state
            self.opponent.move(action)

            self.init_opponent = False
        else:
            # Now, let the model do all the work

            # Update buffers
            self._update_buffers()

            # Current state
            state = State(
                robot_location=self.opponent_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                opponent_location=self.robot_location_buffer.retreive(),
            )

            # Determine next action
            action = self.opponent.get_action(state)

            # Update game state
            self.opponent.move(action)

            # Update buffers
            self._update_buffers()

            # New state
            new_state = State(
                robot_location=self.opponent_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                opponent_location=self.robot_location_buffer.retreive(),
            )

            # Get updated stats
            # stats = self.conn.get("stats")

            # Observation of the game at the moment
            observation = Observation(
                state=state,
                action=action,
                reward=self.data["reward"],  # Opposite reward of our agent, only works for current reward settings
                done=self.data["done"],
                new_state=new_state,
            )

            # Update model
            self.opponent.update(observation)

            # # After so many iterations, save motedel
            if self.iterations % self.iterations_on_save == 0:
                self.opponent.save_model()

        return None

    def run(self) -> None:
        """ Main guts of training """

        # Game loop
        while True:

            # Train for an alotted amount of time
            if time.time() - self.time < self.wait:

                # Our Agent
                self.main_player()

                # Our opponent
                self.opponent_player()

                # Update iterator
                self.iterations += 1

                # Scores
                # self.scores = self.conn.get("scores")

                # Compute scores
                if self.data["opponent_score"] == 10:
                    print(
                        f"Agent {self.data['robot_score']}, Computer {self.data['opponent_score']}"
                    )
                    print("Computer wins!")
                    self.conn.post({"reset": {"total": True}})

                if self.data["robot_score"] == 10:
                    print(
                        f"Agent {self.data['robot_score']}, Computer {self.data['opponent_score']} "
                    )
                    print("Agent wins!")
                    self.conn.post({"reset": {"total": True}})

            else:
                print("Training time elasped")
                sys.exit()


if __name__ == "__main__":
    # Run program
    train = Train()
    train.run()

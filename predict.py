""" Play Air Hockey with learned models """
import argparse
import logging
import os
import sys
from typing import Dict, Union

import numpy as np
from redis import ConnectionError

from connect import RedisConnection
from environment import AirHockey
from rl import MemoryBuffer, Strategy
from utils import Observation, State, write_results

# Initiate logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Predict:
    def __init__(self, args: Dict[str, Union[str, int]]):

        # Set up Redis
        self.redis = RedisConnection()

        # Parse cli args
        self.args = args

        # Load Environment
        self.env = AirHockey()

        # Set up our robot
        self.robot = Strategy().make(
            env=self.env, strategy=self.args["robot"], capacity=self.args["capacity"], train=False
        )
        self.robot.agent_name = "robot"

        # Set up our opponent. The opponent can also be a human player.
        self.opponent = Strategy().make(
            env=self.env, strategy=self.args["opponent"], capacity=self.args["capacity"], train=False
        )
        self.opponent.agent_name = (
            "human" if self.args["opponent"] == "human" else "opponent"
        )

        # Interesting and important constants
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

        # Update buffers
        self._update_buffers()
        logger.info("Connected to Redis")

    def _update_buffers(self) -> None:
        """ Update redis buffers """

        data = self.redis.get()
        self.puck_location = data["puck"]["location"]
        self.robot_location = data["robot"]["location"]
        self.opponent_location = data["opponent"]["location"]

        self.robot_location_buffer.append(tuple(self.robot_location))
        self.puck_location_buffer.append(tuple(self.puck_location))
        self.opponent_location_buffer.append(tuple(self.opponent_location))

        return None

    # Todo - Record stats of training
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
    #             f"Agent Win", self.agent_cumulative_win, self.itqerations
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

    def robot_player(self) -> None:
        """ Main player """

        # For first move, move in a random direction
        if self.init:
            action = np.random.randint(0, 3)

            # Update game state
            self.robot.move(action)
            logger.debug(f"Robot took action: {action}")

            self.init = False
        else:
            # Now, let the model do all the work

            # Update buffers
            self._update_buffers()

            # Current state
            state = State(
                robot_location=self.robot_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
            )

            # Determine next action
            action = self.robot.get_action(state)
            logger.debug(f"Robot took action: {action}")

            # Update game state
            self.robot.move(action)

            # Update buffers
            self._update_buffers()

        return None

    def opponent_player(self) -> None:
        """ Opponent player """

        # Human opponent
        if self.opponent.agent_name == "human":
            self.opponent.move(self.opponent_location)
            return None

        # RL Model opponent
        # For first move, move in a random direction
        if self.init_opponent:

            action = np.random.randint(0, 3)

            # Update game state
            self.opponent.move(action)
            logger.debug(f"Opponent took action: {action}")

            self.init_opponent = False
        else:
            # Now, let the model do all the work

            # Update buffers
            self._update_buffers()

            # Current state
            state = State(
                robot_location=self.opponent_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
            )

            # Determine next action
            action = self.opponent.get_action(state)
            logger.debug(f"Opponent took action: {action}")

            # Update game state
            self.opponent.move(action)

            # Update buffers
            self._update_buffers()

        return None

    def run(self) -> None:
        """ Main guts of training """

        # Game loop
        while True:

            # Our Agent
            self.robot_player()

            # Our opponent
            self.opponent_player()

            # Update iterator
            # self.iterations += 1

            # Compute scores
            if self.env.opponent_score == 10:
                logger.info(
                    f"Agent {self.env.robot_score}, Computer {self.env.opponent_score}"
                )
                logger.info("Computer wins!")
                self.env.reset(total=True)

            if self.env.robot_score == 10:
                logger.info(
                    f"Agent {self.env.robot_score}, Computer {self.env.opponent_score} "
                )
                logger.info("Agent wins!")
                self.env.reset(total=True)


if __name__ == "__main__":
    """ Run predictions """
    parser = argparse.ArgumentParser(description="Play game with model predictions.")

    parser.add_argument("-r", "--robot", help="Robot strategy")
    parser.add_argument("-o", "--opponent", help="Opponent strategy")
    parser.add_argument(
        "-c", "--capacity", default=5, help="Number of past expierences to store"
    )

    parser.add_argument(
        "--tensorboard",
        help="Tensorbaord log location. If none is specified, then Tensorboard will not be used.",
    )
    args = vars(parser.parse_args())

    # Validation
    if not args.get("robot"):
        logger.error("Robot strategy Undefined")
        sys.exit()

    if not args.get("opponent"):
        logger.error("Opponent strategy Undefined")
        sys.exit()

    # Run program
    try:
        predict = Predict(args)
    except ConnectionError:
        logger.error(
            "Cannot connect to Redis. Please make sure Redis is up and active."
        )
        sys.exit()

    predict.run()

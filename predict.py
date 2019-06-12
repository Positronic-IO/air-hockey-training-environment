""" Play Air Hockey with learned models """
import argparse
import logging
import os
import sys
import time
from typing import Dict, Union

import numpy as np
from redis import ConnectionError

from connect import RedisConnection
from environment import AirHockey
from rl import MemoryBuffer, Strategy
from rl.utils import Observation, State

# Initiate logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pygame

pygame.init()


class Predict:
    def __init__(self, args: Dict[str, Union[str, int]]):

        # Set up Redis
        self.redis = RedisConnection()

        # Parse cli args
        self.args = args

        # Load Environment
        self.env = AirHockey()

        # Set up our robot
        self.robot = Strategy().make(env=self.env, strategy=self.args["robot"], train=False)
        self.robot.agent_name = "robot"

        # Set up our opponent. The opponent can also be a human player.
        self.opponent = Strategy().make(env=self.env, strategy=self.args["opponent"], train=False)
        self.opponent.agent_name = "human" if self.args["opponent"] == "human" else "opponent"

        # Interesting and important constants
        self.iterations = 1
        self.new = False

        # We begin..
        self.init, self.init_opponent = True, True

        # Cumulative scores
        self.robot_cumulative_score, self.opponent_cumulative_score = 0, 0

        # Cumulative wins
        self.robot_cumulative_win, self.opponent_cumulative_win = 0, 0

        # Set up buffers for agent position and puck position
        self.robot_location_buffer = MemoryBuffer(self.args["capacity"])
        self.puck_location_buffer = MemoryBuffer(self.args["capacity"])
        self.opponent_location_buffer = MemoryBuffer(self.args["capacity"])

        # Set up buffers for agent velocity and puck velocity
        self.robot_velocity_buffer = MemoryBuffer(self.args["capacity"])
        self.puck_velocity_buffer = MemoryBuffer(self.args["capacity"])
        self.opponent_velocity_buffer = MemoryBuffer(self.args["capacity"])

        # Update buffers
        self._update_buffers()
        logger.info("Connected to Redis")

    def _update_buffers(self) -> None:
        """ Update redis buffers """

        data = self.redis.get()
        puck_location = data["puck"]["location"]
        robot_location = data["robot"]["location"]

        # This is an attribute because it is referenced when there exists a human player.
        self.opponent_location = data["opponent"]["location"]

        puck_velocity = data["puck"]["velocity"]
        robot_velocity = data["robot"]["velocity"]
        opponent_velocity = (0, 0) if self.opponent.agent_name == "human" else data["opponent"]["velocity"]

        self.robot_location_buffer.append(tuple(robot_location))
        self.puck_location_buffer.append(tuple(puck_location))
        self.opponent_location_buffer.append(tuple(self.opponent_location))

        self.robot_velocity_buffer.append(tuple(robot_velocity))
        self.puck_velocity_buffer.append(tuple(puck_velocity))
        self.opponent_velocity_buffer.append(tuple(opponent_velocity))

        return None

    def robot_player(self) -> None:
        """ Main player """

        # For first move, move in a random direction
        if self.init:
            action = np.random.randint(0, 4)

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
                robot_location=self.robot_location_buffer.retreive(average=False),
                puck_location=self.puck_location_buffer.retreive(average=False),
                robot_velocity=self.robot_velocity_buffer.retreive(average=False),
                puck_velocity=self.puck_velocity_buffer.retreive(average=False),
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

            action = np.random.randint(0, 4)

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
                robot_location=self.opponent_location_buffer.retreive(average=False),
                puck_location=self.puck_location_buffer.retreive(average=False),
                robot_velocity=self.opponent_velocity_buffer.retreive(average=False),
                puck_velocity=self.puck_velocity_buffer.retreive(average=False),
            )

            # Determine next action
            action = self.opponent.get_action(state)
            logger.debug(f"Opponent took action: {action}")

            # Update game state
            self.opponent.move(action)

            # Update buffers
            self._update_buffers()

        return None

    def play(self) -> None:
        """ Play a round for training """

        # Our Agent
        self.robot_player()

        # Our opponent
        self.opponent_player()

        # Update iterator
        self.iterations += 1

        # Compute scores
        if self.env.opponent_score == 10:
            logger.info(f"Robot {self.env.robot_score}, Computer {self.env.opponent_score}")
            logger.info("Computer wins!")
            self.env.reset(total=True)

        if self.env.robot_score == 10:
            logger.info(f"Robot {self.env.robot_score}, Computer {self.env.opponent_score} ")
            logger.info("Robot wins!")
            self.env.reset(total=True)

    def run(self) -> None:
        """ Main guts of training """

        # Game loop
        while True:

            # Play a frame
            self.play()

            time.sleep(1 / self.args["fps"])


if __name__ == "__main__":
    """ Run predictions """
    parser = argparse.ArgumentParser(description="Play game with model predictions.")

    parser.add_argument("-r", "--robot", help="Robot strategy")
    parser.add_argument("-o", "--opponent", help="Opponent strategy")
    parser.add_argument("-c", "--capacity", default=1, help="Number of past expierences to store")
    parser.add_argument("--fps", default=60, help="Frame per second")
    args = vars(parser.parse_args())

    # Valid
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
        logger.error("Cannot connect to Redis. Please make sure Redis is up and active.")
        sys.exit()

    predict.run()

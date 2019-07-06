""" Play Air Hockey with learned models """
import argparse
import logging
import os
import sys
import time
from typing import Dict, Union

import numpy as np
import redis

from environment import AirHockey
from lib.connect import RedisConnection
from lib.strategy import Strategy
from lib.types import Observation, State

# Initiate logger
logging.basicConfig(level=logging.INFO)
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
        self.robot = Strategy.make(env=self.env, strategy=self.args["robot"], train=False)
        self.robot.name = "robot"

        # Set up our opponent. The opponent can also be a human player.
        self.opponent = Strategy.make(env=self.env, strategy=self.args["opponent"], train=False)
        self.opponent.name = "human" if self.args["opponent"] == "human" else "opponent"

        # We begin..
        self.init, self.init_opponent = True, True

        # Cumulative scores
        self.robot_cumulative_score, self.opponent_cumulative_score = 0, 0

        # Cumulative wins
        self.robot_cumulative_win, self.opponent_cumulative_win = 0, 0

        # Update buffers
        self._update_buffers()
        logger.info("Connected to Redis")

    def _update_buffers(self) -> None:
        """ Update redis buffers """

        data = self.redis.get("components")["components"]
        self.puck_location = data["puck"]["location"]
        self.robot_location = data["robot"]["location"]

        # Pull from browser instead of pygame
        if self.args["opponent"] == "human":
            _opponent_location = self.redis.get("new-opponent-location")
            self.opponent_location = tuple(
                [_opponent_location["new-opponent-location"]["x"], _opponent_location["new-opponent-location"]["y"]]
            )
        else:
            self.opponent_location = data["opponent"]["location"]

        self.puck_velocity = data["puck"]["velocity"]
        self.robot_velocity = data["robot"]["velocity"]
        self.opponent_velocity = (0, 0) if self.opponent.name == "human" else data["opponent"]["velocity"]

        return None

    def robot_player(self) -> None:
        """ Main player """

        # Update buffers
        self._update_buffers()

        # Determine next action
        action = self.robot.get_action()
        logger.debug(f"Robot took action: {action}")

        # Update game state
        self.robot.move(action)

        # Take a new step in the MDP
        score, _ = self.robot.step(action)

        # Update environment if the action we took was one that scores
        self.env.update_score(score)

        # Update buffers
        self._update_buffers()

    def opponent_player(self) -> None:
        """ Opponent player """

        # Human opponent
        if self.opponent.name == "human":
            self.opponent.move(self.opponent_location)
            score = 0
            if self.env.puck in self.env.left_goal:
                score = -1

            if self.env.puck in self.env.right_goal:
                score = 1

            # Update environment if the action we took was one that scores
            self.env.update_score(score)
            return None

        # ---RL opponent----

        # Update buffers
        self._update_buffers()

        # Determine next action
        action = self.opponent.get_action()
        logger.debug(f"Opponent took action: {action}")

        # Update game state
        self.opponent.move(action)

        # Take a new step in the MDP
        score, _ = self.opponent.step(action)

        # Update environment if the action we took was one that scores
        self.env.update_score(score)

        # Update buffers
        self._update_buffers()

    def play(self) -> None:
        """ Play a round for training """

        # Our Agent
        self.robot_player()

        # Our opponent
        self.opponent_player()

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

            # Alert the positions are different
            self.redis.publish("position-update")

            # Play a frame
            self.play()

            if int(self.args["fps"]) > 0:
                time.sleep(1 / int(self.args["fps"]))


if __name__ == "__main__":
    """ Run predictions """
    parser = argparse.ArgumentParser(description="Play game with model predictions.")

    parser.add_argument("-r", "--robot", help="Robot strategy")
    parser.add_argument("-o", "--opponent", help="Opponent strategy")
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
    except redis.ConnectionError:
        logger.error("Cannot connect to Redis. Please make sure Redis is up and active.")
        sys.exit()

    predict.run()

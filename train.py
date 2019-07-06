""" Air Hockey Training Simulator """
import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, Union

import numpy as np
import redis
from pytz import timezone

from environment import AirHockey
from lib.strategy import Strategy
from lib.connect import RedisConnection
from lib.types import Observation, State
from lib.utils.io import record_data_csv, record_model_info

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Train:
    def __init__(self, args: Dict[str, Union[str, int]]):

        # Parse cli args
        self.args = args

        # Set up Redis
        self.redis = RedisConnection()

        # Load Environment
        self.env = AirHockey()

        # Set up our robot
        self.robot = Strategy.make(env=self.env, strategy=self.args["robot"], train=True)
        self.robot.name = "robot"

        # # Set up our opponent. The opponent can also be a human player.
        self.opponent = Strategy.make(env=self.env, strategy=self.args["opponent"], train=True)
        self.opponent.name = "human" if self.args["opponent"] == "human" else "opponent"

        # Save model architectures with an unique run id
        robot_path, opponent_path, counter = record_model_info(self.args["robot"], self.args["opponent"])

        # Paths to save models
        self.robot.save_path = robot_path
        if self.opponent.name != "human":
            self.opponent.save_path = opponent_path

        # We begin..
        self.init, self.init_opponent = True, True

        # Cumulative scores
        self.robot_cumulative_score, self.opponent_cumulative_score = 0, 0

        # Cumulative wins
        self.robot_cumulative_win, self.opponent_cumulative_win = 0, 0

        # Update buffers
        self._update_buffers()
        logger.info("Connected to Redis")

        # Initial time
        self.time = time.time()
        self.wait = (60 ** 2) * float(self.args["time"])  # Defaults to 3 hours
        logger.info(f"Training time: {self.args['time']} hours")

        # Useful when saving to csv
        self.env.stats_dir = str(counter)

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

    def stats(self) -> None:
        """ Record training stats """

        results = dict()

        results = {
            "created_at": datetime.now(timezone("America/Chicago")),
            "robot_goal": 0,
            "opponent_goal": 0,
            "robot_win": 0,
            "opponent_win": 0,
        }

        if self.env.robot_score > self.robot_cumulative_score:
            results["robot_goal"] = 1
            self.robot_cumulative_score += 1

        if self.env.opponent_score > self.opponent_cumulative_score:
            results["opponent_goal"] = 1
            self.opponent_cumulative_score += 1

        if self.env.robot_score == 10:
            results["robot_win"] = 1
            self.robot_cumulative_score = 0
            self.opponent_cumulative_score = 0

        if self.env.opponent_score == 10:
            results["opponent_win"] = 1
            self.robot_cumulative_score = 0
            self.opponent_cumulative_score = 0

        # Save to csv
        record_data_csv(self.env.stats_dir, "scores", results)

    def robot_player(self) -> None:
        """ Main player """

        # For first move, move in a random direction
        if self.init:

            # Continuous actions
            if getattr(self.robot, "continuous", False):
                action = np.random.uniform(-3, 3), np.random.uniform(-3, 3)
            else:
                # Disrete actions
                action = np.random.randint(0, 4)

            # Update game state
            self.robot.move(action)

            self.init = False
        else:
            # Now, let the model do all the work

            # Update buffers
            self._update_buffers()

            # Determine next action
            action = self.robot.get_action()

            # Update game state
            self.robot.move(action)

            # Take a new step in the MDP
            score, observation = self.robot.step(action)

            # Update buffers
            self._update_buffers()
            
            # Update environment if the action we took was one that scores
            self.env.update_score(score)

            # Save stats to CSV
            self.stats()

        # Save stats to CSV
        self.stats()

        return None

    def opponent_player(self) -> None:
        """ Opponent player """

        # If the opponent is human
        if self.opponent.name == "human":
            self.opponent.move(self.opponent_location)
            return None

        # For first move, move in a random direction
        if self.init_opponent:

            # Continuous actions
            if getattr(self.opponent, "continuous", False):
                action = np.random.uniform(-3, 3), np.random.uniform(-3, 3)
            else:
                # Disrete actions
                action = np.random.randint(0, 4)

            # Update game state
            self.opponent.move(action)

            self.init_opponent = False
        else:
            # Now, let the model do all the work

            # Update buffers
            self._update_buffers()

            # Determine next action
            action = self.opponent.get_action()

            # Update game state
            self.opponent.move(action)

            # Take a new step in the MDP
            score, observation = self.opponent.step(action)

            # Update buffers
            self._update_buffers()

            # Update environment if the action we took was one that scores
            self.env.update_score(score)

            # Save stats to CSV
            self.stats()

        return None

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

            # Train for an alotted amount of time
            if time.time() - self.time < self.wait:

                # Alert the positions are different
                self.redis.publish("position-update")

                # Play a frame
                self.play()

            else:
                logger.info("Training time elasped")
                sys.exit()


if __name__ == "__main__":
    """ Start Training """
    parser = argparse.ArgumentParser(description="Process stuff for training.")

    parser.add_argument("-r", "--robot", help="Robot strategy")
    parser.add_argument("-o", "--opponent", help="Opponent strategy")
    parser.add_argument("-t", "--time", default=3, help="Time per train. Units in hours. (Default to 3 hours)")
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
        train = Train(args)
    except redis.ConnectionError:
        logger.error("Cannot connect to Redis. Please make sure Redis is up and active.")
        sys.exit()

    train.run()

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
from pymongo import MongoClient
from redis import ConnectionError

from connect import RedisConnection
from environment import AirHockey
from rl import MemoryBuffer, Strategy
from utils import Observation, State, write_results

# Initiate Logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Train:
    def __init__(self, args: Dict[str, Union[str, int]]):

        # Parse cli args
        self.args = args

        # Set up Redis
        self.redis = RedisConnection()
        self.mongo = MongoClient()["stats"]

        # Load Environment
        self.env = AirHockey()

        # Set up our robot
        self.robot = Strategy().make(
            env=self.env,
            strategy=self.args["robot"],
            capacity=self.args["capacity"],
            train=True,
        )
        self.robot.agent_name = "robot"

        # Set up our opponent. The opponent can also be a human player.
        self.opponent = Strategy().make(
            env=self.env,
            strategy=self.args["opponent"],
            capacity=self.args["capacity"],
            train=True,
        )
        self.opponent.agent_name = (
            "human" if self.args["opponent"] == "human" else "opponent"
        )

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

        # Update buffers
        self._update_buffers()
        logger.info("Connected to Redis")

        # Initial time
        self.time = time.time()
        self.wait = (60 ** 2) * int(self.args["time"])  # Defaults to 3 hours
        logger.info(f"Training time: {self.args['time']} hours")

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

    def stats(self) -> None:
        """ Record training stats """

        results = dict()

        results = {
            "created_at": datetime.utcnow(),
            "robot_goal": False,
            "opponent_goal": False,
            "robot_win": False,
            "opponent_win": False,
        }

        if self.env.reward == self.env.rewards["point"]:
            results["robot_goal"] = True

        if self.env.reward == self.env.rewards["loss"]:
            results["opponent_goal"] = True

        if self.env.robot_score == 10:
            results["robot_win"] = True

        if self.env.opponent_score == 10:
            results["opponent_win"] = True

        self.mongo[self.args["stats"]].insert(results)

        return None

    def robot_player(self) -> None:
        """ Main player """

        # For first move, move in a random direction
        if self.init:
            action = np.random.randint(0, 4)

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
                reward=self.env.reward,
                done=self.env.done,
                new_state=new_state,
            )

            # Update model
            self.robot.update(observation)

        # Save stats to Mongo
        self.stats()

        # After so many iterations, save model
        if self.iterations % self.iterations_on_save == 0:
            self.robot.save_model()

        return None

    def opponent_player(self) -> None:
        """ Opponent player """

        # If the opponent is human
        if self.opponent.agent_name == "human":
            self.opponent.move(self.opponent_location)
            return None

        # For first move, move in a random direction
        if self.init_opponent:

            action = np.random.randint(0, 4)

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

            # Observation of the game at the moment
            observation = Observation(
                state=state,
                action=action,
                reward=self.env.reward,  # Opposite reward of our agent, only works for current reward settings
                done=self.env.done,
                new_state=new_state,
            )

            # Update model
            self.opponent.update(observation)

            # Save stats to Mongo
            self.stats()

            # # After so many iterations, save motedel
            if self.iterations % self.iterations_on_save == 0:
                self.opponent.save_model()

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
            logger.info(
                f"Robot {self.env.robot_score}, Computer {self.env.opponent_score}"
            )
            logger.info("Computer wins!")
            self.env.reset(total=True)

        if self.env.robot_score == 10:
            logger.info(
                f"Robot {self.env.robot_score}, Computer {self.env.opponent_score} "
            )
            logger.info("Robot wins!")
            self.env.reset(total=True)

    def run(self) -> None:
        """ Main guts of training """

        # Game loop
        while True:

            # Train for an alotted amount of time
            if time.time() - self.time < self.wait:

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
    parser.add_argument(
        "-c", "--capacity", default=5, help="Number of past expierences to store"
    )
    parser.add_argument(
        "-s", "--stats", default="test", help="MongoDB to store training data"
    )
    parser.add_argument(
        "-t",
        "--time",
        default=3,
        help="Time per train. Units in hours. (Default to 3 hours)",
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
        train = Train(args)
    except ConnectionError:
        logger.error(
            "Cannot connect to Redis. Please make sure Redis is up and active."
        )
        sys.exit()

    train.run()

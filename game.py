""" Air Hockey Training Simulator """
import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict, Tuple, Union

import numpy as np
import redis
from pytz import timezone

from environment import AirHockey
from lib.agents import Agent
from lib.strategy import Strategy
from lib.types import Observation, State
from lib.utils.connect import RedisConnection
from lib.utils.io import get_runid, record_data, record_data_csv

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Game:
    def __init__(self, args: Dict[str, Union[str, int, bool]]):

        # Parse cli args
        self.args = args

        # Set up Redis
        self.redis = RedisConnection()

        # Load Environment
        self.env = AirHockey()

        runid, path = get_runid(os.path.join("/", "data", "air-hockey", "output"))
        logger.info(f"Run id: {runid}")
        logger.info(f"Run path: {path}")

        os.environ["LOAD_RUN"] = self.args.get("run")  # Past run to load and use
        os.environ["PROJECT"] = path

        # Set up our robot
        self.robot = Strategy.make(env=self.env, strategy=self.args.get("model"), train=self.args.get("play"))
        self.robot.name = "robot"

        # Set up our opponent. The opponent can also be a human player.
        if self.args.get("human"):
            self.opponent = Agent(self.env)
            self.opponent.name = "human"

        # Save model architectures and rewards with an unique run id
        record_data(self.args.get("model"))

        # We begin..
        self.init = True

        # Cumulative scores, Cumulative wins
        self.robot_cumulative_score, self.opponent_cumulative_score = 0, 0
        self.robot_cumulative_win, self.opponent_cumulative_win = 0, 0

        # Initial time
        if self.args.get("play"):
            logger.info(f"Game mode: Train")
            self.time = time.time()
            self.wait = (60 ** 2) * float(self.args.get("time"))  # Defaults to 3 hours
            logger.info(f"Training time: {self.args.get('time')} hours")

    @property
    def human_location(self) -> Tuple[int, int]:
        """ Get human location from redis """

        retval = self.redis.get("new-opponent-location")
        location = tuple([retval["new-opponent-location"]["x"], retval["new-opponent-location"]["y"]])
        return location

    def stats(self) -> None:
        """ Record training stats """

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
        record_data_csv("scores", results)

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
            # Determine next action
            action = self.robot.get_action()

            # Update game state
            self.robot.move(action)

            # Take a new step in the MDP
            score, observation = self.robot.step(action)

            # Update environment if the action we took was one that scores
            self.env.update_score(score)

            # Save stats to CSV
            self.stats()

        return None

    def human_player(self) -> None:
        """ Opponent player """

        # If the opponent is human
        self.opponent.move(self.human_location)
        return None

    def play(self) -> None:
        """ Play a round for training """

        # Our Agent
        self.robot_player()

        # Our opponent
        if self.args.get("human"):
            self.human_player()
        else:
            self.env.update_state(agent_name="computer")

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
            if self.args.get("train") and time.time() - self.time > self.wait:
                logger.info("Training time elasped")
                sys.exit()

            # Alert the positions are different
            self.redis.publish("position-update")

            # Play a frame
            self.play()

            if self.args.get("fps") > 0:
                time.sleep(1 / self.args.get("fps"))


if __name__ == "__main__":
    """ Start Training """
    parser = argparse.ArgumentParser(description="Process stuff for training.")

    parser.add_argument("--model", type=str, help="Robot strategy")
    parser.add_argument("--run", type=str, default="", help="Specific past run for robot strategy to use")
    parser.add_argument("--time", default=3, type=float, help="Time per train. Units in hours. (Default to 3 hours)")
    parser.add_argument("--fps", default=-1, type=int, help="Frame per second")
    parser.add_argument("--human", action="store_true", help="Human players")
    parser.add_argument("--play", action="store_false", help="Play game instead of train")
    args = vars(parser.parse_args())

    # Validation
    if not args.get("model"):
        logger.error("Robot strategy Undefined")
        sys.exit()

    # Run program
    try:
        game = Game(args)
    except redis.ConnectionError:
        logger.error("Cannot connect to Redis. Please make sure Redis is up and active.")
        sys.exit()

    game.run()

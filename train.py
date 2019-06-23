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
from pymongo.errors import ServerSelectionTimeoutError
from redis import ConnectionError

from connect import RedisConnection
from environment import AirHockey
from rl import MemoryBuffer, Strategy
from rl.utils import Observation, State, record_model_info

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
        self.mongo = MongoClient()
        self.mongo_error = False

        try:
            # Test if connection exists
            self.mongo.server_info()
        except ServerSelectionTimeoutError:
            self.mongo_error = True

        # Load Environment
        self.env = AirHockey()

        # Set up our robot
        self.robot = Strategy().make(env=self.env, strategy=self.args["robot"], train=True)
        self.robot.agent_name = "robot"

        # # Set up our opponent. The opponent can also be a human player.
        self.opponent = Strategy().make(env=self.env, strategy=self.args["opponent"], train=True)
        self.opponent.agent_name = "human" if self.args["opponent"] == "human" else "opponent"

        # Save model architectures with an unique run id
        robot_path, opponent_path = record_model_info(self.args["robot"], self.args["opponent"])

        # Paths to save models
        self.robot.save_path = robot_path
        if self.opponent.agent_name != "human":
            self.opponent.save_path = opponent_path

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

        # Initial time
        self.time = time.time()
        self.wait = (60 ** 2) * float(self.args["time"])  # Defaults to 3 hours
        logger.info(f"Training time: {self.args['time']} hours")

    def _update_buffers(self) -> None:
        """ Update redis buffers """

        data = self.redis.get("components")["components"]
        puck_location = data["puck"]["location"]
        robot_location = data["robot"]["location"]

        # Pull from browser instead of pygame
        if self.args["opponent"] == "human":
            _opponent_location = self.redis.get("new-opponent-location")
            self.opponent_location = tuple(
                [_opponent_location["new-opponent-location"]["x"], _opponent_location["new-opponent-location"]["y"]]
            )
        else:
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

        if self.env.robot_score > self.robot_cumulative_score:
            results["robot_goal"] = True
            self.robot_cumulative_score += 1

        if self.env.opponent_score > self.opponent_cumulative_score:
            results["opponent_goal"] = True
            self.opponent_cumulative_score += 1

        if self.env.robot_score == 10:
            results["robot_win"] = True
            self.robot_cumulative_score = 0
            self.opponent_cumulative_score = 0

        if self.env.opponent_score == 10:
            results["opponent_win"] = True
            self.robot_cumulative_score = 0
            self.opponent_cumulative_score = 0

        # Test if connection exists
        if not self.mongo_error:
            self.mongo["stats"][self.args["stats"]].insert(results)

        return None

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

            # Current state
            state = State(
                robot_location=self.robot_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                robot_velocity=self.robot_velocity_buffer.retreive(),
                puck_velocity=self.puck_velocity_buffer.retreive(),
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
                robot_velocity=self.robot_velocity_buffer.retreive(),
                puck_velocity=self.puck_velocity_buffer.retreive(),
            )

            # Observation of the game at the moment
            observation = Observation(
                state=state, action=action, reward=self.env.robot_reward, done=self.env.robot_done, new_state=new_state
            )

            # Update model
            self.robot.update(observation)

        # Save stats to Mongo
        self.stats()

        return None

    def opponent_player(self) -> None:
        """ Opponent player """

        # If the opponent is human
        if self.opponent.agent_name == "human":
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

            # Current state
            state = State(
                robot_location=self.opponent_location_buffer.retreive(),
                puck_location=self.puck_location_buffer.retreive(),
                robot_velocity=self.opponent_velocity_buffer.retreive(),
                puck_velocity=self.puck_velocity_buffer.retreive(),
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
                robot_velocity=self.opponent_velocity_buffer.retreive(),
                puck_velocity=self.puck_velocity_buffer.retreive(),
            )

            # Observation of the game at the moment
            observation = Observation(
                state=state,
                action=action,
                reward=self.env.opponent_reward,  # Opposite reward of our agent, only works for current reward settings
                done=self.env.opponent_done,
                new_state=new_state,
            )

            # Update model
            self.opponent.update(observation)

            # Save stats to Mongo
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
    except ConnectionError:
        logger.error("Cannot connect to Redis. Please make sure Redis is up and active.")
        sys.exit()

    train.run()

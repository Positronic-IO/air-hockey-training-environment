""" Air Hockey Game Environment """

import json
import logging
from time import time
from typing import Any, Dict, Tuple, Union

import numpy as np
from redis import Redis

from connect import RedisConnection
from environment.components import Goal, Mallet, Puck, Table
from rl import RewardTracker
from rl.utils import Action, Observation, State

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AirHockey:

    # Default rewwards
    rewards = {"point": 1, "loss": -1}

    def __init__(self) -> None:
        """ Initiate an air hockey game """

        # Set up Redis Connection
        self.redis = RedisConnection()

        # Some constants for air hockey components
        width_offset = (27, 40)
        x_offset = 38
        y_offset = 50

        # Create Table
        self.table = Table(size=(900, 480), x_offset=x_offset, width_offset=width_offset)

        # Make goals
        self.left_goal = Goal(x=0, y=self.table.midpoints[1] - y_offset, w=width_offset[0])
        self.right_goal = Goal(x=self.table.size[0] - x_offset, y=self.table.midpoints[1] - y_offset, w=width_offset[1])

        # Puck settings
        puck_radius = 15

        # Create puck
        self.puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1], radius=puck_radius, redis=self.redis)

        # Define left and right mallet positions
        mallet_l = self.table.midpoints[0] - 100, self.table.midpoints[1]
        mallet_r = self.table.midpoints[0] + 100, self.table.midpoints[1]

        # Makes Computer Mallet
        self.robot = Mallet(
            "robot",
            mallet_l[0],
            mallet_l[1],
            right_lim=self.table.midpoints[0] - puck_radius,
            table_size=self.table.size,
            redis=self.redis,
        )

        # Makes Computer Mallet
        self.opponent = Mallet(
            "opponent",
            mallet_r[0],
            mallet_r[1],
            left_lim=self.table.midpoints[0] + puck_radius,
            table_size=self.table.size,
            redis=self.redis,
        )

        # Default scores
        self.opponent_score = 0
        self.robot_score = 0

        # Push to redis
        self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})

        # Physics
        self.ticks_to_friction = 60
        self.ticks_to_ai = 10

        # Define step size of mallet
        self.step_size = 10

        # Set timer for stalling
        self.timer = time()

        # Reward
        self.robot_reward, self.opponent_reward = 0, 0

        # If episode is done
        self.robot_done, self.opponent_done = False, False

        # Reward trackers
        self.robot_reward_tracker = RewardTracker("robot", self.left_goal, self.right_goal, self.table)
        self.opponent_reward_tracker = RewardTracker("opponent", self.left_goal, self.right_goal, self.table)

    def _move(self, agent: Mallet, action: Action) -> None:
        """ Move agent's mallet """

        # Update action
        if isinstance(action, (tuple, list)):  # Cartesian Coordinates
            agent.dx, agent.dy = action[0], action[1]

        # Integers
        if isinstance(action, int) and action == 0:
            agent.y += self.step_size

        if isinstance(action, int) and action == 1:
            agent.y += -self.step_size

        if isinstance(action, int) and action == 2:
            agent.x += -self.step_size

        if isinstance(action, int) and action == 3:
            agent.x += self.step_size

        # Set agent position
        agent.update_mallet()

        # Update agent velocity
        agent.dx = agent.x - agent.last_x
        agent.dy = agent.y - agent.last_y
        agent.update_mallet()

        return None

    def update_state(self, action: Action, agent_name: str = "robot") -> None:
        """ Update state of game """

        # Move mallet
        if agent_name == "robot":
            self._move(self.robot, action)
        elif agent_name == "opponent":
            self._move(self.opponent, action)
        elif agent_name == "human":
            # Update action
            if isinstance(action, tuple) or isinstance(action, list):  # Cartesian Coordinates
                self.opponent.x, self.opponent.y = action[0], action[1]
                self.opponent.update_mallet()
        else:
            logger.error("Invalid agent name")
            raise ValueError

        # Determine puck physics
        # If the mallet is in the neighborhood of the puck, do stuff.
        if self.puck & self.robot and self.puck | self.robot:
            self.puck.dx = -3 * self.puck.dx + self.robot.dx
            self.puck.dy = -3 * self.puck.dy + self.robot.dy

        if self.puck & self.opponent and self.puck | self.opponent:
            self.puck.dx = -3 * self.puck.dx + self.opponent.dx
            self.puck.dy = -3 * self.puck.dy + self.opponent.dy

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update_puck()

        # Update puck position
        while self.ticks_to_friction == 0:
            self.puck.friction_on_puck()
            self.ticks_to_friction = 60

        # Update agent position
        self.robot.last_x = self.robot.x
        self.robot.last_y = self.robot.y
        self.robot.update_mallet()

        self.opponent.last_x = self.opponent.x
        self.opponent.last_y = self.opponent.y
        self.opponent.update_mallet()

        # Update score
        self.update_score()

        self.ticks_to_friction -= 1
        self.ticks_to_ai -= 1

        return None

    def update_score(self) -> Union[int, None]:
        """ Get current score """

        # Get reward stats from reward trackers
        self.robot_reward, robot_scored, self.robot_done = self.robot_reward_tracker(self.puck, self.robot)
        self.opponent_reward, opponent_scored, self.opponent_done = self.opponent_reward_tracker(
            self.puck, self.opponent
        )

        # When then agent scores on the computer
        if robot_scored > 0:
            self.robot_score += 1

            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})

            logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.reset()
            return None

        # When the computer scores on the agent
        if opponent_scored > 0:
            self.opponent_score += 1

            # Push to redis
            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})

            logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.reset()
            return None

        return None

    def reset(self, total: bool = False) -> None:
        """ Reset Game """

        if total:
            self.opponent_score = 0
            self.robot_score = 0

            # Push to redis
            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})
            logger.info("Total Game reset")

        self.puck.reset()
        self.robot.reset_mallet()
        self.opponent.reset_mallet()

        return None

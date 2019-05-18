""" Air Hockey Game Environment """

import json
import logging
from time import time
from typing import Any, Dict, Tuple, Union

import numpy as np
from redis import Redis

from connect import RedisConnection
from environment.components import Goal, Mallet, Puck
from utils import Action, Observation, State, gaussian

# Initiate Logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AirHockey:

    # Default rewwards
    rewards = {"point": 1, "loss": -1}

    def __init__(self, **kwargs) -> None:
        """ Initiate an air hockey game """

        self.redis = RedisConnection()

        # Define table and rink sizes
        self.table_size = kwargs.get("table_size", [900, 480])
        self.rink_size = kwargs.get("rink_size", [840, 440])

        # Create board
        self.board = np.zeros(shape=self.table_size, dtype=int)

        # Puck settings
        self.puck_radius = kwargs.get("puck_radius", 15)

        # Default scores
        self.opponent_score = 0
        self.robot_score = 0

        # Push to redis
        self.redis.post(
            {
                "scores": {
                    "robot_score": self.robot_score,
                    "opponent_score": self.opponent_score,
                }
            }
        )

        self.ticks_to_friction = 60
        self.ticks_to_ai = 10

        # Define midpoints
        self.table_midpoints = list(map(lambda x: int(x / 2), self.table_size))

        # Define left and right mallet positions
        default_left_position = self.table_midpoints[0] - 100, self.table_midpoints[1]
        default_right_position = self.table_midpoints[0] + 100, self.table_midpoints[1]

        # Set puck initial position
        puck_start_x, puck_start_y = self.table_midpoints[0], self.table_midpoints[1]

        # Create puck
        self.puck = Puck(x=puck_start_x, y=puck_start_y, redis=self.redis)

        # Make goals
        self.left_goal = Goal(0, self.table_midpoints[1] - 50, w=27)
        self.right_goal = Goal(
            self.table_size[0] - 38, self.table_midpoints[1] - 50, w=40
        )

        # Makes Computer Mallet
        self.robot = Mallet(
            "robot",
            default_left_position[0],
            default_left_position[1],
            right_lim=self.table_midpoints[0] - self.puck_radius,
            table_size=self.table_size,
            redis=self.redis,
        )

        # Makes Computer Mallet
        self.opponent = Mallet(
            "opponent",
            default_right_position[0],
            default_right_position[1],
            left_lim=self.table_midpoints[0] + self.puck_radius,
            table_size=self.table_size,
            redis=self.redis,
        )

        # Define step size of mallet
        self.step_size = 10

        # Set timer for stalling
        self.timer = time()

        # Reward
        self.reward = 0

        # If episode is done
        self.done = False

    def _move(self, agent: Mallet, action: Action) -> None:
        """ Move agent's mallet """

        # Update action
        if isinstance(action, tuple) or isinstance(
            action, list
        ):  # Cartesian Coordinates
            agent.x, agent.y = action[0], action[1]

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
            if isinstance(action, tuple) or isinstance(
                action, list
            ):  # Cartesian Coordinates
                self.opponent.x, self.opponent.y = action[0], action[1]
                self.opponent.update_mallet()
        else:
            logger.error("Invalid agent name")
            raise ValueError

        # Determine puck physics
        if (
            abs(self.robot.x - self.puck.x) <= 50
            and abs(self.robot.y - self.puck.y) <= 50
        ):
            self.puck.dx = -3 * self.puck.dx + self.robot.dx
            self.puck.dy = -3 * self.puck.dy + self.robot.dy

        if (
            abs(self.opponent.x - self.puck.x) <= 50
            and abs(self.opponent.y - self.puck.y) <= 50
        ):
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

        # When then agent scores on the computer
        if (
            abs(self.right_goal.centre_y - self.puck.y) <= 45
            and abs(self.right_goal.centre_x - self.puck.x) <= 45
        ):
            self.robot_score += 1
            self.reward = self.rewards["point"]

            # Push to redis
            self.redis.post(
                {
                    "scores": {
                        "robot_score": self.robot_score,
                        "opponent_score": self.opponent_score,
                    }
                }
            )

            logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.done = True
            self.reset()
            return None

        # When the computer scores on the agent
        if (
            abs(self.left_goal.centre_y - self.puck.y) <= 45
            and abs(self.left_goal.centre_x - self.puck.x) <= 45
        ):
            self.opponent_score += 1
            self.reward = self.rewards["loss"]

            # Push to redis
            self.redis.post(
                {
                    "scores": {
                        "robot_score": self.robot_score,
                        "opponent_score": self.opponent_score,
                    }
                }
            )

            logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.done = True
            self.reset()
            return None

        if abs(self.right_goal.centre_x - self.puck.x) <= 60:
            # Puck hit the opponet's wall
            self.reward = gaussian(self.puck.y, self.right_goal.centre_y, 50)
            self.done = False
            return None

        if abs(self.left_goal.x - self.puck.x) <= 60:
            # Puck hit the robot's wall
            self.reward = -1 * gaussian(self.puck.y, self.left_goal.centre_y, 50)
            self.done = False
            return None

        # If nothing happens
        self.done = False
        self.reward = 0
        return None

    def reset(self, total: bool = False) -> None:
        """ Reset Game """

        if total:
            self.opponent_score = 0
            self.robot_score = 0

            # Push to redis
            self.redis.post(
                {
                    "scores": {
                        "robot_score": self.robot_score,
                        "opponent_score": self.opponent_score,
                    }
                }
            )
            logger.info("Total Game reset")

        self.puck.reset()
        self.robot.reset_mallet()
        self.opponent.reset_mallet()

        return None

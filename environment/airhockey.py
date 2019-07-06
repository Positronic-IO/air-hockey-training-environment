""" Air Hockey Game Environment """
import json
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
from redis import Redis

from environment import config
from environment.goal import Goal
from environment.mallet import Mallet
from environment.puck import Puck
from environment.table import Table
from lib.connect import RedisConnection
from lib.types import Action, Observation, State

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AirHockey:
    def __init__(self) -> None:
        """ Initiate an air hockey game """

        # Set up Redis Connection
        self.redis = RedisConnection()

        # Directory to save csv data
        self.stats_dir = ""

        # Create Table
        self.table = Table()

        # Make goals
        self.left_goal = Goal(x=0, y=self.table.midpoints[1])
        self.right_goal = Goal(x=self.table.size[0], y=self.table.midpoints[1])

        # Create puck
        self.puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1])

        # Define left and right mallet positions
        mallet_l = self.table.midpoints[0] - 100, self.table.midpoints[1]
        mallet_r = self.table.midpoints[0] + 100, self.table.midpoints[1]

        # Makes Robot Mallet
        self.robot = Mallet(name="robot", x=mallet_l[0], y=mallet_l[1])

        # Makes Computer Mallet
        self.opponent = Mallet(name="opponent", x=mallet_r[0], y=mallet_r[1])

        # Default scores
        self.opponent_score, self.robot_score = 0, 0

        # Push to redis
        # self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})

        # Drift
        self.ticks_to_friction = 60

        # Define step size of mallet
        self.step_size = 10

    def _move(self, agent: "Mallet", action: "Action") -> None:
        """ Move agent's mallet """

        # Contiunous action
        if isinstance(action, (tuple, list)):  # Cartesian Coordinates
            agent.dx += action[0]
            agent.dy += action[1]

        # Discrete action
        if isinstance(action, int) and action == 0:
            agent.y += self.step_size

        if isinstance(action, int) and action == 1:
            agent.y += -self.step_size

        if isinstance(action, int) and action == 2:
            agent.x += -self.step_size

        if isinstance(action, int) and action == 3:
            agent.x += self.step_size

        # Set agent position
        agent.update()

        # Update agent velocity
        agent.dx = agent.x - agent.last_x
        agent.dy = agent.y - agent.last_y
        agent.update()

        return None

    def update_state(self, action: "Action", agent_name: str = "robot") -> None:
        """ Update state of game """

        # Move mallet
        if agent_name == "robot":
            self._move(self.robot, action)
        elif agent_name == "opponent":
            self._move(self.opponent, action)
        elif agent_name == "human":
            # Update action
            if isinstance(action, (tuple, list)):  # Cartesian Coordinates
                self.opponent.x, self.opponent.y = action[0], action[1]
                self.opponent.update()
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

        # Update agent and oppponent positions
        self.robot.update()
        self.opponent.update()

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update()

        # Update puck position
        while self.ticks_to_friction == 0:
            self.puck.friction_on_puck()
            self.ticks_to_friction = 60
        self.ticks_to_friction -= 1

        # Update Redis
        self.redis.post(
            {
                "components": {
                    "puck": {"location": self.puck.location(), "velocity": self.puck.velocity()},
                    self.robot.name: {"location": self.robot.location(), "velocity": self.robot.velocity()},
                    self.opponent.name: {"location": self.opponent.location(), "velocity": self.opponent.velocity()},
                }
            }
        )

        return None

    # TODO - Add acceleration?
    def get_state(self, agent_name: str = "robot"):
        """ Get current state """

        agent = self.robot if agent_name == "robot" else self.opponent
        state = State(
            agent_location=agent.location(),
            puck_location=self.puck.location(),
            agent_velocity=agent.velocity(),
            puck_velocity=self.puck.velocity(),
        )

        return state

    def update_score(self, score: int) -> Union[int, None]:
        """ Get current score """
        # When then agent scores on the computer
        if score > 0:
            self.robot_score += 1
            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})
            self.redis.publish("score-update")
            logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.reset()
            return None

        # When the computer scores on the agent
        if score < 0:
            self.opponent_score += 1
            # Push to redis
            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})
            self.redis.publish("score-update")
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
            self.redis.publish("score-update")
            logger.info("Total Game reset")

        self.puck.reset()
        self.robot.reset()
        self.opponent.reset()

        return None

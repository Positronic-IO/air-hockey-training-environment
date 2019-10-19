""" Air Hockey Game Environment """
import json
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import config
from environment.goal import Goal
from environment.heuristic import computer
from environment.mallet import Mallet
from environment.physics import collision
from environment.puck import Puck
from environment.table import Table
from lib.types import Action, Observation, State
from lib.utils.exceptions import InvalidAgentError

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AirHockey:
    def __init__(self) -> None:
        """ Initiate an air hockey game """

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

        self.mallets = [self.robot, self.opponent]

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

    def update_state(self, action: "Action" = "", agent_name: str = "") -> None:
        """ Update state of game """

        # Move mallet
        if agent_name == "robot":
            self._move(self.robot, action)
        elif agent_name == "human":
            # Update action
            self.opponent.x, self.opponent.y = action[0], action[1]
            self.opponent.update()
        elif agent_name == "computer":  # Non-human opponent
            computer(self.puck, self.opponent)
        else:
            logger.error("Invalid agent name")
            raise InvalidAgentError

        # Check for collisions, do physics magic, update objects
        for mallet in self.mallets:
            collision(self.puck, mallet)

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update()

        # Update agent and oppponent positions
        for mallet in self.mallets:
            mallet.update()

        return None

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

        # When the computer scores on the agent
        if score < 0:
            self.opponent_score += 1

        logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
        self.reset()

    def reset(self, total: bool = False) -> None:
        """ Reset Game """

        if total:
            logger.info("Total Game reset")
            self.opponent_score = 0
            self.robot_score = 0

        self.puck.reset()
        self.robot.reset()
        self.opponent.reset()

        return None

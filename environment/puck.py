""" Puck object """
import json
from typing import Any, Tuple, Union

import numpy as np

from environment.table import Table
from environment import config
from environment.goal import Goal
from environment.mallet import Mallet


class Puck:
    """ Puck object """

    def __init__(self, x: int, y: int, dx: int = -3, dy: int = 0):
        """ Create a goal """

        self.name = "puck"
        self.radius = config.puck["radius"]

        # Puck position
        self.x = x
        self.y = y

        # Last puck position
        self.last_x = x
        self.last_y = y

        # Puck velocity
        self.dx = dx  # Default is -3. This implies the robot get the puck first.
        self.dy = dy

        # Mass
        self.mass = config.puck["mass"]
        self.imass = 1.0 / self.mass  # Inverse mass

        # Initial puck position
        self.puck_start_x = self.x
        self.puck_start_y = self.y

        # Default puck speed
        self.default_speed = config.puck["default_speed"]

    def update(self) -> None:
        """ Update puck position """

        # Enforces puck stays inside the table
        if self.x <= self.radius:
            self.x = self.radius
            self.dx *= -1
        elif self.x >= config.table["size"][0] - self.radius:
            self.x = config.table["size"][0] - self.radius
            self.dx *= -1

        if self.y <= self.radius:
            self.y = self.radius
            self.dy *= -1
        elif self.y >= config.table["size"][1] - self.radius:
            self.y = config.table["size"][1] - self.radius
            self.dy *= -1

        # Record last known position (within the constraints of the table)
        self.last_x = self.x
        self.last_y = self.y

        # Update positions
        self.x += self.dx
        self.y += self.dy

        return None

    def friction_on_puck(self) -> None:
        """ Define friction on puck, mimic real life to some extent """

        # Horizontal
        if self.dx > 1:
            self.dx -= 0.5
        if self.dx < -1:
            self.dx += 0.5

        # Vertical
        if self.dy > 1:
            self.dy -= 0.5
        elif self.dy < -1:
            self.dy += 0.5

        return None

    def limit_puck_speed(self) -> None:
        """ Limit speed of puck """

        # Horizontal
        if self.dx > self.default_speed:
            self.dx = self.default_speed
        if self.dx < -self.default_speed:
            self.dx = -self.default_speed

        # Vertical
        if self.dy > self.default_speed:
            self.dy = self.default_speed
        if self.dy < -self.default_speed:
            self.dy = self.default_speed

        # Record last known position (within the constraints of the table)
        self.last_x = self.x
        self.last_y = self.y

        return None

    def reset(self) -> None:
        """ Rest puck to a ranfom initial position, makes sure AI does learn a fast start """
        self.x = self.puck_start_x
        self.y = self.puck_start_y
        self.dx = np.random.uniform(-10, 10)
        self.dy = np.random.uniform(-10, 10)

        return None

    def location(self) -> Tuple[int, int]:
        """ Cartesian coordinates """

        return int(self.x), int(self.y)

    def prev_location(self) -> Tuple[int, int]:
        """ Previous location """

        return int(self.last_x), int(self.last_y)

    def velocity(self) -> Tuple[int, int]:
        """ Velocity of Puck """

        return self.dx, self.dy

    def __and__(self, mallet: "Mallet") -> bool:
        """ Determine if the puck and other objects overlap """

        distance = np.sqrt((self.x - mallet.x) ** 2 + (self.y - mallet.y) ** 2)
        radius = self.radius + mallet.radius

        # Check to see if there is any intersection
        if distance < radius:
            return True

        # No intersection
        return False

    def __lshift__(self, table: Table) -> bool:
        """ Determine if the puck and the left side of table overlap """

        if abs(self.x - table.left_wall) < self.radius:
            return True

        # No intersection
        return False

    def __rshift__(self, table: Table) -> bool:
        """ Determine if the puck and the right side of table overlap """

        if abs(self.x - table.right_wall) < self.radius:
            return True

        # No intersection
        return False

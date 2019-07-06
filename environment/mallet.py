""" Mallet Component """
import json
import random
from typing import Tuple

from environment import config
from environment.table import Table


class Mallet:
    def __init__(self, name: str, x: int, y: int, dx: int = 0, dy: int = 0, **kwargs):
        """ Create a mallet """

        # Define mallet type
        self.name = name

        # Set current position (default to initial position
        self.x, self.y = x, y

        # Save current state for later
        self.last_x = self.x
        self.last_y = self.y

        # Physical properties
        self.mass = config.mallet["mass"]
        self.radius = config.mallet["radius"]

        # Define table size
        self.table = Table()

        # Set horizontal and vertical limits
        self.u_lim = self.radius
        self.b_lim = self.table.size[1] - self.radius

        # Midline of table
        if self.name == "robot":
            self.left_lim = self.radius
            self.right_lim = self.table.midpoints[0] - self.radius
        elif self.name == "opponent":
            self.left_lim = self.table.midpoints[0] + self.radius
            self.right_lim = self.table.size[0] - self.radius
        else:
            self.left_lim, self.right_lim = (self.radius, self.table.size[0] - self.radius)

        # Set default velocity
        self.dx = dx
        self.dy = dx

        # Set mallet position
        # self.mallet_start_x = self.x
        # self.mallet_start_y = self.y

    def update(self) -> None:
        """ Update mallet position """

        # Save current state for later
        self.last_x = self.x
        self.last_y = self.y

        # if self.name != "agent":
        self.x += self.dx
        self.y += self.dy

        # Enforce mallet to be in table
        if self.x < self.left_lim:
            self.x = self.left_lim
        elif self.x > self.right_lim:
            self.x = self.right_lim

        if self.y < self.u_lim:
            self.y = self.u_lim
        elif self.y > self.b_lim:
            self.y = self.b_lim

        return None

    def reset(self) -> None:
        """ Reset mallet """

        self.dx = 0
        self.dy = 0

        return None

    def location(self) -> Tuple[int, int]:
        """ Current Cartesian coordinates """

        return int(self.x), int(self.y)

    def prev_location(self) -> Tuple[int, int]:
        """ Previous location """

        return int(self.last_x), int(self.last_y)

    def velocity(self) -> Tuple[int, int]:
        """ Velocity of Puck """

        return self.dx, self.dy

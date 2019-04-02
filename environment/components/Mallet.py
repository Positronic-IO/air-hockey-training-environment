""" Mallet Component """
import random
from typing import Tuple


class Mallet(object):
    def __init__(self, name: str, x: int, y: int, **kwargs):
        """ Create a mallet """

        # Set current position (default to initial position
        self.x, self.y = x, y

        # Save current state for later
        self.last_x = self.x
        self.last_y = self.y

        # Define mallet type
        self.name = name

        # Define table size
        table_size = kwargs["table_size"]

        # Set horizontal and vertical limits
        self.u_lim = kwargs.get("u_lim", 47)
        self.b_lim = kwargs.get("b_lim", table_size[1] - 40)
        self.left_lim = kwargs.get("left_lim", 47)
        self.right_lim = kwargs.get("right_lim", table_size[0] - 57)

        # Set default velocity
        self.dx = kwargs.get("dx", 0)
        self.dy = kwargs.get("dy", 0)

        # Set mallet position
        self.mallet_start_x = self.x
        self.mallet_start_y = self.y

    def update_mallet(self) -> None:
        """ Update mallet position """
        if self.name != "agent":
            self.x += self.dx
            self.y += self.dy

        # Enforce mallet to be in table
        if self.x < self.left_lim:
            self.dx = 0
            self.x = self.left_lim
        elif self.x > self.right_lim:
            self.dx = 0
            self.x = self.right_lim

        if self.y < self.u_lim:
            self.dy = 0
            self.y = self.u_lim
        elif self.y > self.b_lim:
            self.dy = 0
            self.y = self.b_lim

        return None

    def reset_mallet(self) -> None:
        """ Reset mallet """

        self.x = self.mallet_start_x
        self.y = self.mallet_start_y

        return None

    def location(self) -> Tuple[int, int]:
        """ Cartesian coordinates """

        return self.x, self.y

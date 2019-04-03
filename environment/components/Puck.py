import random
from typing import Tuple


class Puck(object):
    """ Puck object """

    def __init__(self, x: int, y: int, dx: int = -5, dy: int = 3):
        """ Create a goal """

        self.name = "puck"

        # Puck position
        self.x = x
        self.y = y

        # Last puck position
        self.last_x = x
        self.last_y = y

        # Puck velocity
        self.dx = dx
        self.dy = dy

        # Initial puck position
        self.puck_start_x = self.x
        self.puck_start_y = self.y

        # Default puck speed
        self.puck_speed = 10

    def update_puck(self) -> None:
        """ Update puck position """

        # Enforces puck stays inside the table
        if self.x <= 47:
            self.x = 47
            self.dx *= -1
        elif self.x >= 850:
            self.x = 850
            self.dx *= -1

        if self.y <= 40:
            self.y = 42
            self.dy *= -1
        elif self.y >= 460:
            self.y = 458
            self.dy *= -1
        
        # Record last known position (within the constraints of the table)
        self.last_x = self.x
        self.last_y = self.y

        # Update position
        self.x += self.dx
        self.y += self.dy

        return None

    def friction_on_puck(self) -> None:
        """ Define friction on puck, mimic real life to some extent """

        # Horizontal
        if self.dx > 1:
            self.dx -= 1
        elif self.dx < -1:
            self.dx += 1

        # Vertical
        if self.dy > 1:
            self.dy -= 1
        elif self.dy < -1:
            self.dy += 1

        return None

    def limit_puck_speed(self) -> None:
        """ Limit speed of puck """

        # Horizontal
        if self.dx > 10:
            self.dx = self.puck_speed
        if self.dx < -10:
            self.dx = -self.puck_speed

        # Vertical
        if self.dy > 10:
            self.dy = self.puck_speed
        if self.dy < -10:
            self.dy = self.puck_speed

        # Record last known position (within the constraints of the table)
        self.last_x = self.x
        self.last_y = self.y

        return None

    def reset(self) -> None:
        """ Rest puck to initial position """
        self.x = self.puck_start_x
        self.y = self.puck_start_y
        self.dx = -5
        self.dy = 3

        return None

    def location(self) -> Tuple[int, int]:
        """ Cartesian coordinates """

        return self.x, self.y

    def prev_location(self) -> Tuple[int, int]:
        """ Previous location """

        return self.last_x, self.last_y
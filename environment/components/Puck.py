""" Puck object """
import json
import numpy as np
from typing import Tuple, Any

from environment.components import Table


class Puck:
    """ Puck object """

    def __init__(self, x: int, y: int, dx: int = -5, dy: int = 3, radius: int = 0):
        """ Create a goal """

        self.name = "puck"
        self.radius = radius

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
        """ Rest puck to a ranfom initial position, makes sure AI does learn a fast start """
        self.x = self.puck_start_x
        self.y = self.puck_start_y
        self.dx = np.random.uniform(-5, 5)
        self.dy = np.random.uniform(-5, 5)

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

    def __and__(self, component: Any) -> bool:
        """ Determine if the puck and other objects overlap """

        if component.__class__.__name__.lower() == "table":
            # Check to see if there is any intersection in the x-axis
            if abs(self.x - component.left_wall) <= 50 or abs(component.right_wall - self.x) <= 50:
                return True
        else:
            # Check to see if there is any intersection in the x-axis
            if abs(self.x - component.x) <= 50:
                return True

        # No intersection
        return False

    def __or__(self, component: Any) -> bool:
        """ Determine if the puck and other objects overlap (y-axis) """

        if abs(self.y - component.y) <= 50:
            return True

        # No intersection
        return False

    def __lshift__(self, component: Table) -> bool:
        """ Determine if the puck and the left side of table overlap """

        if abs(self.x - component.left_wall) <= 50:
            return True

        # No intersection
        return False

    def __rshift__(self, component: Table) -> bool:
        """ Determine if the puck and the right side of table overlap """

        if abs(self.x - component.right_wall) <= 50:
            return True

        # No intersection
        return False

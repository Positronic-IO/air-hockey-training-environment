""" Goal Component """
from typing import Any


class Goal:
    def __init__(self, x: int, y: int):
        """ Create goal """

        # Define center of goal
        self.x = x
        self.y = y

    def __contains__(self, puck: Any) -> bool:

        if abs(self.x - puck.x) < 2 * puck.radius and abs(self.y - puck.y) < 95:
            return True

        return False

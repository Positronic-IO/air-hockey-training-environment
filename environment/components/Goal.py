""" Goal Component """
from typing import Tuple


class Goal(object):
    def __init__(self, x: int, y: int, w: int = 20, h: int = 100):
        """ Create goal """

        #  Initial position
        self.x = x
        self.y = y

        # Width and heighth of goal
        self.w = w
        self.h = h

        # Define center of goal
        self.centre_x = int(self.x + self.w / 2)
        self.centre_y = int(self.y + self.h / 2)

    def location(self) -> Tuple[int, int]:
        """ Cartesian coordinates """

        return self.centre_x, self.centre_y

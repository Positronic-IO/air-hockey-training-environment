""" Goal Component """
from typing import Tuple


class Goal(object):
    def __init__(self, x: int, y: int, w: int = 20, h: int = 100):
        """ Create goal """

        # Define center of goal
        self.left_corner_x, self.left_corner_y = x, y
        self.w, self.h = w, h

        self.x = int(self.left_corner_x + self.w / 2)
        self.y = int(self.left_corner_y + self.h / 2)


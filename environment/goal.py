""" Goal Component """
from typing import Tuple

from environment import config


class Goal:
    def __init__(self, x: int, y: int):
        """ Create goal """

        # Define center of goal
        self.left_corner_x, self.left_corner_y = x, y
        self.w, self.h = config.goal["w"], config.goal["h"]

        self.x = int(self.left_corner_x + self.w / 2)
        self.y = int(self.left_corner_y + self.h / 2)

""" Table Component """
from typing import Tuple

from environment import config


class Table:
    def __init__(self):
        """ Create table """

        self.size = config.table["size"]
        self.midpoints = tuple(map(lambda x: int(x / 2), self.size))

        self.left_wall = int(config.table["width_offset"][0] / 2)
        self.right_wall = int(self.size[0] - config.table["x_offset"] + config.table["width_offset"][1] / 2)

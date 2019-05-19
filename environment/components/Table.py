""" Table Component """
from typing import Tuple


class Table(object):
    def __init__(
        self,
        size: Tuple[int, int] = (900, 480),
        width_offset: Tuple[int, int] = (27, 40),
        x_offset: int = 38,
    ):
        """ Create goal """

        self.size = size
        self.midpoints = tuple(map(lambda x: int(x / 2), self.size))

        self.left_wall = int(width_offset[0] / 2)
        self.right_wall = int(self.size[0] - x_offset + width_offset[1] / 2)

""" Test Air Hockey Game Environment """

from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from environment.components import Goal, Mallet
from environment.components.test import TestPuck


class TestAirHockey(AirHockey):
    def __init__(self, **kwargs) -> None:
        """ Initiate an test air hockey game environment"""

        super().__init__(**kwargs)
        self.puck = TestPuck()
        rewards = {"point": 100, "loss": -200, "hit": 200, "miss": -500}

    def get_reward(self) -> int:
        """ Get reward of the current action """

        # Drive mallet away from corners
        if (
            self.left_mallet.x >= self.left_mallet.right_lim
            or self.left_mallet.x <= self.left_mallet.left_lim
            or self.left_mallet.y >= self.left_mallet.u_lim
            or self.left_mallet.y <= self.left_mallet.b_lim
        ):

            return self.rewards["miss"]

        # We hit the puck
        if (
            abs(self.left_mallet.x - self.puck.x) <= 35
            and abs(self.left_mallet.y - self.puck.y) <= 35
        ):
            # self.reset()
            return self.rewards["hit"]

        # We missed the puck
        return self.rewards["miss"]

    def update_state(self, action: Union[str, Tuple[int, int]]) -> None:
        """ Update state of game """

        # Update action
        if isinstance(action, tuple):
            self.left_mallet.x, self.left_mallet.y = action[0], action[1]

        if action == self.actions[0]:
            self.left_mallet.y += 10

        if action == self.actions[1]:
            self.left_mallet.y += -10

        if action == self.actions[2]:
            self.left_mallet.x += 10

        if action == self.actions[3]:
            self.left_mallet.x += -10

        # Set agent position
        self.left_mallet.update_mallet()

        # Update agent velocity
        self.left_mallet.dx = self.left_mallet.x - self.left_mallet.last_x
        self.left_mallet.dy = self.left_mallet.y - self.left_mallet.last_y

        # Determine puck physics
        if (
            abs(self.left_mallet.x - self.puck.x) <= 35
            and abs(self.left_mallet.y - self.puck.y) <= 35
        ):
            self.puck.dx = -1 * self.puck.dx + self.left_mallet.dx
            self.puck.dy = -1 * self.puck.dy + self.left_mallet.dy

        if (
            abs(self.right_mallet.x - self.puck.x) <= 35
            and abs(self.right_mallet.y - self.puck.y) <= 35
        ):
            self.puck.dx = -1 * self.puck.dx + self.right_mallet.dx
            self.puck.dy = -1 * self.puck.dy + self.right_mallet.dy

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update_puck()

        self.right_mallet.update_mallet()

        # Update agent position
        self.left_mallet.last_x = self.left_mallet.x
        self.left_mallet.last_y = self.left_mallet.y


        self.ticks_to_friction -= 1
        self.ticks_to_ai -= 1

        return None


""" Test Air Hockey Game Environment """

from typing import Any, Dict, Tuple, Union

import numpy as np

from environment import AirHockey
from environment.components import Goal, Mallet
from environment.test import TestPuck


class TestAirHockey(AirHockey):
    def __init__(self, **kwargs) -> None:
        """ Initiate an test air hockey game environment"""

        super().__init__(**kwargs)
        self.puck = TestPuck()

    def get_reward(self) -> int:
        """ Get reward of the current action """

        # Drive mallet away from corners
        if (
            self.agent.x >= self.agent.right_lim
            or self.agent.x <= self.agent.left_lim
            or self.agent.y >= self.agent.u_lim
            or self.agent.y <= self.agent.b_lim
        ):

            return self.rewards["miss"]

        # We hit the puck
        if (
            abs(self.agent.x - self.puck.x) <= 35
            and abs(self.agent.y - self.puck.y) <= 35
        ):
            # self.reset()
            return self.rewards["hit"]

        # We missed the puck
        return self.rewards["miss"]

    def update_state(self, action: Union[str, Tuple[int, int]]) -> None:
        """ Update state of game """

        # Update action
        if isinstance(action, tuple):
            self.agent.x, self.agent.y = action[0], action[1]

        if action == self.actions[0]:
            self.agent.y += 10

        if action == self.actions[1]:
            self.agent.y += -10

        if action == self.actions[2]:
            self.agent.x += 10

        if action == self.actions[3]:
            self.agent.x += -10

        # Set agent position
        self.agent.update_mallet()

        # Update agent velocity
        self.agent.dx = self.agent.x - self.agent.last_x
        self.agent.dy = self.agent.y - self.agent.last_y

        # Determine puck physics
        if (
            abs(self.agent.x - self.puck.x) <= 35
            and abs(self.agent.y - self.puck.y) <= 35
        ):
            self.puck.dx = -1 * self.puck.dx + self.agent.dx
            self.puck.dy = -1 * self.puck.dy + self.agent.dy

        if (
            abs(self.opponent.x - self.puck.x) <= 35
            and abs(self.opponent.y - self.puck.y) <= 35
        ):
            self.puck.dx = -1 * self.puck.dx + self.opponent.dx
            self.puck.dy = -1 * self.puck.dy + self.opponent.dy

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update_puck()

        self.opponent.update_mallet()

        # Update agent position
        self.agent.last_x = self.agent.x
        self.agent.last_y = self.agent.y


        self.ticks_to_friction -= 1
        self.ticks_to_ai -= 1

        return None


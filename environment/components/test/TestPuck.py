""" Puck for testing """
import random
from environment.components import Puck


class TestPuck(Puck):
    def __init__(self):
        super().__init__(x=100, y=100, dx=6, dy=6)

    def update_puck(self) -> None:
        """ Update puck position """

        # Update velocity

        if self.x <= 47:
            self.x = 47
            self.dx *= -1
        elif self.x >= 850:
            self.x = 850
            self.dx *= -1
        elif self.x >= 460:
            self.x = 458
            self.dx *= -1

        if self.y <= 40:
            self.y = 42
            self.dy *= -1
        elif self.y >= 460:
            self.y = 458
            self.dy *= -1

        # Update position
        self.x += self.dx
        self.y += self.dy

        return None

    def reset(self) -> None:
        """ Rest puck to initial position """
        self.x = self.puck_start_x
        self.y = self.puck_start_y
        self.dx = random.randint(-30, 30)
        self.dy = random.randint(-30, 30)

        return None

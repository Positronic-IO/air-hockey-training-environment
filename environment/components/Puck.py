import random

class Puck(object):
    """ Puck object """

    def __init__(self, x: int, y: int, dx: int= 0, dy: int= 0):
        """ Create a goal """

        self.name = "puck"

        # Puck Position
        self.x = x
        self.y = y

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

        # Update velocity

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

        return None

    def reset(self) -> None:
        """ Rest puck to initial position """
        self.x = self.puck_start_x
        self.y = self.puck_start_y
        self.dx = 0
        self.dy = 0

        return None

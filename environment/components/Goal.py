""" Goal Comaponent """


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
        self.centre_x = self.x + self.w / 2
        self.centre_y = self.y + self.h / 2

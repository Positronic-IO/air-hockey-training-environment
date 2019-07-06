""" Test Puck Object """
from environment import Goal, Mallet, Puck, Table


class TestPuck:
    def setup(self):
        # Create Table
        self.table = Table()

        # Make goals
        self.left_goal = Goal(x=0, y=self.table.midpoints[1])
        self.right_goal = Goal(x=self.table.size[0], y=self.table.midpoints[1])

        # Define left and right mallet positions
        self.mallet_l = self.table.midpoints[0] - 100, self.table.midpoints[1]
        self.mallet_r = self.table.midpoints[0] + 100, self.table.midpoints[1]

    def test_puck_intersect_mallet(self):
        """ Test if puck intersects a mallet """

        # Create puck
        puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1])

        # Makes a Mallet
        mallet = Mallet(
            "mallet", self.mallet_l[0], self.mallet_l[1], right_lim=self.table.midpoints[0], table_size=self.table.size
        )

        # Intersect puck
        puck.x, puck.y = self.mallet_l[0] - 10, self.mallet_l[1] - 10
        assert puck & mallet

        # Does not intersect
        puck.x, puck.y = self.mallet_l[0] - 50, self.mallet_l[1] - 50
        assert not puck & mallet

    def test_puck_intersect_left_wall(self):
        """ Test if puck hits the left wall (excluding goal) """

        # Create puck
        puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1])

        # Intersect (along x-axis)
        puck.x, puck.y = 10, 300
        assert puck << self.table

        # Does not intersect (along x-axis)
        puck.x, puck.y = 100, 300
        assert not puck << self.table

    def test_puck_intersect_right_wall(self):
        """ Test if puck hits the right wall (excluding goal) """

        # Create puck
        puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1])

        # Intersect (along x-axis)
        puck.x, puck.y = self.table.size[0] - 10, 300
        assert puck >> self.table

        # Does not intersect (along x-axis)
        puck.x, puck.y = self.table.size[0] - 100, 300
        assert not puck >> self.table

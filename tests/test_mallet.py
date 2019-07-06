""" Test Mallet Object """
from environment import Goal, Mallet, Puck, Table


class TestPuck:
    def setup(self):
        # Create Table
        self.table = Table()

        # Define left and right mallet positions
        self.mallet_l = self.table.midpoints[0] - 100, self.table.midpoints[1]
        self.mallet_r = self.table.midpoints[0] + 100, self.table.midpoints[1]

    def test_mallet_stays_in_table(self):
        """ Test to make sure mallet stays in table """

        mallet = Mallet("mallet", self.mallet_l[0], self.mallet_l[1])

        # Left and upper limits
        mallet.x, mallet.y = -1, -1
        mallet.update()
        assert mallet.x == mallet.radius
        assert mallet.y == mallet.radius

        # Right and bottom limits
        mallet.x, mallet.y = 1000, 1000
        mallet.update()
        assert mallet.x == self.table.size[0] - mallet.radius
        assert mallet.y == self.table.size[1] - mallet.radius

    def test_robot_mallet_table(self):
        """ Test to make sure robot's mallet stays on the left side of the table """

        mallet = Mallet("robot", self.mallet_l[0], self.mallet_l[1])

        # Left and upper limits
        mallet.x, mallet.y = -1, -1
        mallet.update()
        assert mallet.x == mallet.radius
        assert mallet.y == mallet.radius

        # Right and bottom limits
        mallet.x, mallet.y = 1000, 1000
        mallet.update()
        assert mallet.x == self.table.midpoints[0] - mallet.radius
        assert mallet.y == self.table.size[1] - mallet.radius

    def test_opponent_mallet_table(self):
        """ Test to make sure opponent's mallet stays on the right side of the table """

        mallet = Mallet("opponent", self.mallet_r[0], self.mallet_r[1])

        # Left and upper limits
        mallet.x, mallet.y = -1, -1
        mallet.update()
        assert mallet.x == self.table.midpoints[0] + mallet.radius
        assert mallet.y == mallet.radius

        # Right and bottom limits
        mallet.x, mallet.y = 1000, 1000
        mallet.update()
        assert mallet.x == self.table.size[0] - mallet.radius
        assert mallet.y == self.table.size[1] - mallet.radius

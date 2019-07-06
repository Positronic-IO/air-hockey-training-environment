""" Test goal """
from environment import Goal, Puck, Table


class TestGoal:
    def setup(self):
        # Create Table
        self.table = Table()

        # Make goals
        self.left_goal = Goal(x=0, y=self.table.midpoints[1])
        self.right_goal = Goal(x=self.table.size[0], y=self.table.midpoints[1])

    def test_puck_intersect_left_goal(self):
        """ Test if puck is in the left goal """

        # Create puck
        puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1])

        # Puck in goal
        puck.x, puck.y = 10, self.left_goal.y
        assert puck in self.left_goal

        # Puck not in goal
        puck.x, puck.y = 10, self.left_goal.y - 210
        assert not (puck in self.left_goal)

    def test_puck_intersect_right_goal(self):
        """ Test if puck is in the right goal """

        # Create puck
        puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1])

        # Puck in goal
        puck.x, puck.y = self.right_goal.x, self.right_goal.y
        assert puck in self.right_goal

        # Puck not in goal
        puck.x, puck.y = self.right_goal.x, self.right_goal.y - 210
        assert not (puck in self.right_goal)

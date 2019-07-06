""" Test Environment """
from environment import AirHockey


class TestEnvironment:
    def setup(self):
        pass

    def test_puck_update_location(self):
        env = AirHockey()
        # Some actions
        env.update_state((80, 100), "robot")
        env.update_state((350, 233), "opponent")
        env.update_state((200, 234), "robot")
        env.update_state((380, 234), "opponent")
        assert env.puck.location() == (438, 240)

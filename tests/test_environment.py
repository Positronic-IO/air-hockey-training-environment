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
        assert env.puck.location() == (430, 252)

    def test_agent_score(self):
        env = AirHockey()
        env.puck.x, env.puck.y = 867, 225
        env.update_score()
        assert env.agent_score == 1

    def test_cpu_score(self):
        env = AirHockey()
        env.puck.x, env.puck.y = 29, 225
        env.update_score()
        assert env.cpu_score == 1

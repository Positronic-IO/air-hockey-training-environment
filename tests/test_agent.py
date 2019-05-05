from environment import AirHockey
from rl.Agent import Agent


class TestAgent:
    def setup(self):
        pass

    def test_move(self):
        """ Test location of agent """

        env = AirHockey()
        agent = Agent(env)

        prev_location = agent.location()
        new_location = (prev_location[0] + 10, prev_location[0] + 10)
        agent.move(new_location)
        assert agent.location() == new_location

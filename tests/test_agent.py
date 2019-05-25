from environment import AirHockey
from lib import Agent


class TestAgent:
    def setup(self):
        pass

    def test_move_cartesian(self):
        """ Test location of agent """

        env = AirHockey()
        agent = Agent(env)
        agent.agent_name = "robot"

        prev_location = agent.location()
        new_location = (prev_location[0] + 10, prev_location[1] + 10)
        agent.move(new_location)
        assert agent.location() == new_location

    def test_move_up(self):
        """ Test move agent up """

        env = AirHockey()
        agent = Agent(env)
        agent.agent_name = "robot"
        prev_location = agent.location()
        direction = (prev_location[0], prev_location[1] + env.step_size)
        agent.move(0)
        assert agent.location() == direction

    def test_move_down(self):
        """ Test move agent down """

        env = AirHockey()
        agent = Agent(env)
        agent.agent_name = "robot"
        prev_location = agent.location()
        direction = (prev_location[0], prev_location[1] - env.step_size)
        agent.move(1)
        assert agent.location() == direction

    def test_move_left(self):
        """ Test move agent right """

        env = AirHockey()
        agent = Agent(env)
        agent.agent_name = "robot"
        prev_location = agent.location()
        direction = (prev_location[0] - env.step_size, prev_location[1])
        agent.move(2)
        assert agent.location() == direction

    def test_move_right(self):
        """ Test move agent right """

        env = AirHockey()
        agent = Agent(env)
        agent.agent_name = "robot"
        prev_location = agent.location()
        direction = (prev_location[0] + env.step_size, prev_location[1])
        agent.move(3)
        assert agent.location() == direction

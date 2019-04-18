""" Air Hockey Game Environment """

import json
from time import time
from typing import Any, Dict, Tuple, Union

import numpy as np
from redis import Redis

from environment.components import Goal, Mallet, Puck
from utils import Action


class AirHockey:

    redis = Redis()

    # Possible actions
    actions = ["U", "D", "L", "R"]

    # Default rewwards
    rewards = {"point": 3, "loss": -2, "hit": 1, "miss": -0.5}

    def __init__(self, **kwargs) -> None:
        """ Initiate an air hockey game """

        # Define table and rink sizes
        self.table_size = kwargs.get("table_size", [900, 480])
        self.rink_size = kwargs.get("rink_size", [840, 440])

        # Create board
        self.board = np.zeros(shape=self.table_size, dtype=int)

        # Puck settings
        self.puck_radius = kwargs.get("puck_radius", 15)

        # Default scores
        self.cpu_score = 0
        self.agent_score = 0
        self.redis.set(
            "scores",
            json.dumps({"agent_score": self.agent_score, "cpu_score": self.cpu_score}),
        )

        self.ticks_to_friction = 60
        self.ticks_to_ai = 10

        # Define midpoints
        self.table_midpoints = list(map(lambda x: int(x / 2), self.table_size))

        # Define left and right mallet positions
        default_left_position = self.table_midpoints[0] - 100, self.table_midpoints[1]
        default_right_position = self.table_midpoints[0] + 100, self.table_midpoints[1]

        # Set puck initial position
        puck_start_x, puck_start_y = self.table_midpoints[0], self.table_midpoints[1]

        # Create puck
        self.puck = Puck(puck_start_x, puck_start_y)

        # Make goals
        self.left_goal = Goal(0, self.table_midpoints[1] - 50, w=27)
        self.right_goal = Goal(
            self.table_size[0] - 38, self.table_midpoints[1] - 50, w=40
        )

        # Makes Computer Mallet
        self.agent = Mallet(
            "agent",
            default_left_position[0],
            default_left_position[1],
            right_lim=self.table_midpoints[0] - self.puck_radius,
            table_size=self.table_size,
        )

        # Makes Computer Mallet
        self.opponent = Mallet(
            "opponent",
            default_right_position[0],
            default_right_position[1],
            left_lim=self.table_midpoints[0] + self.puck_radius,
            table_size=self.table_size,
        )

        # Define step size of mallet
        self.step_size = 1

        # Set timer for stalling
        self.timer = time()

        # Agent velocity momentum for
        self.momentum = 2

        # Cumulative reward
        self.cumulative_reward = 0

        # Cumulative scores
        self.agent_cumulative_score = 0
        self.cpu_cumulative_score = 0

    def check_stall(self) -> None:
        """ Check to see if the game has stalled """

        if self.puck.x < self.table_midpoints[0]:
            self.timer = time()
            return None

        delta = time() - self.timer

        if (delta > 3) and (self.puck.x > self.table_midpoints[0]):
            self.reset()
            self.timer = time()
            print("Stalled")

        return None

    def reward(self) -> int:
        """ Get reward of the current action """

        # We won, the opponent is a failure
        if self.update_score() == 1:
            return self.rewards["point"]

        # If we lost
        if self.update_score() == -1:
            return self.rewards["loss"]

        # We hit the puck
        if (
            abs(self.agent.x - self.puck.x) <= 35
            and abs(self.agent.y - self.puck.y) <= 35
        ):
            return self.rewards["hit"]

        return self.rewards["miss"]

    # TODO - Have the opponent be its own process (for more complex opponents in the future)
    def malletAI(self, mallet: Mallet) -> None:
        """ The 'AI' of the computer """

        self.check_stall()

        if self.puck.x < mallet.x:
            if self.puck.x < mallet.left_lim:
                mallet.dx = 1
            else:
                mallet.dx = -2

        if self.puck.x > mallet.x:
            if self.puck.x > mallet.right_lim:
                mallet.dx = -1
            else:
                mallet.dx = 2

        if self.puck.y < mallet.y:
            if self.puck.y < mallet.u_lim:
                mallet.dy = 1
            else:
                mallet.dy = -6

        if self.puck.y > mallet.y:
            if self.puck.y <= 360:  # was 250
                mallet.dy = 6
            # elif puck.y<=350:
            #    agent.dy = 2
            else:
                if mallet.y > 200:
                    mallet.dy = -2
                else:
                    mallet.dy = 0
            # Addresses situation when the puck and the computer are on top of each other.
            # Breaks loop
            if abs(self.puck.y - mallet.y) < 40 and abs(self.puck.x - mallet.x) < 40:
                self.puck.dx += 2
                self.puck.dy += 2

        return None

    def update_state(self, action: Action) -> None:
        """ Update state of game """

        # Update action
        if isinstance(action, tuple):  # Cartesian Coordinates
            self.agent.x, self.agent.y = action[0], action[1]

        if isinstance(action, str) and action == self.actions[0]:
            self.agent.y += self.step_size

        if isinstance(action, str) and action == self.actions[1]:
            self.agent.y += -self.step_size

        if isinstance(action, str) and action == self.actions[2]:
            self.agent.x += self.step_size

        if isinstance(action, str) and action == self.actions[3]:
            self.agent.x += -self.step_size

        if self.puck.x < self.agent.x:
            self.agent.dx -= self.momentum

        # Set agent position
        self.agent.update_mallet()

        # Update agent velocity
        self.agent.dx = self.agent.x - self.agent.last_x
        self.agent.dy = self.agent.y - self.agent.last_y
        self.agent.update_mallet()

        # Computer makes its move
        while self.ticks_to_ai == 0:
            self.malletAI(self.opponent)
            self.ticks_to_ai = 10
        self.opponent.update_mallet()

        # Determine puck physics
        if (
            abs(self.agent.x - self.puck.x) <= 35
            and abs(self.agent.y - self.puck.y) <= 35
        ):
            self.puck.dx = -1 * self.puck.dx + self.agent.dx
            self.puck.dy = -1 * self.puck.dy + self.agent.dy

        if (
            abs(self.opponent.x - self.puck.x) <= 35
            and abs(self.opponent.y - self.puck.y) <= 35
        ):
            self.puck.dx = -1 * self.puck.dx + self.opponent.dx
            self.puck.dy = -1 * self.puck.dy + self.opponent.dy

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update_puck()

        # Update puck position
        while self.ticks_to_friction == 0:
            self.puck.friction_on_puck()
            self.ticks_to_friction = 60

        # Update agent position
        self.agent.last_x = self.agent.x
        self.agent.last_y = self.agent.y
        self.agent.update_mallet()

        # Update score
        self.update_score()

        self.ticks_to_friction -= 1
        self.ticks_to_ai -= 1

        return None

    def update_score(self) -> Union[int, None]:
        """ Get current score """

        # When then agent scores on the computer
        if (
            abs(self.right_goal.centre_y - self.puck.y) <= 50
            and abs(self.right_goal.centre_x - self.puck.x) <= 45
        ):
            self.agent_score += 1
            self.agent_cumulative_score += 1

            # Push to redis
            self.redis.set(
                "scores",
                json.dumps(
                    {"agent_score": self.agent_score, "cpu_score": self.cpu_score}
                ),
            )

            print(f"Computer {self.cpu_score}, Agent {self.agent_score}")
            self.reset()
            return 1

        # When the computer scores on the agent
        if (
            abs(self.left_goal.centre_y - self.puck.y) <= 50
            and abs(self.left_goal.centre_x - self.puck.x) <= 45
        ):
            self.cpu_score += 1
            self.cpu_cumulative_score += 1

            # Push to redis
            self.redis.set(
                "scores",
                json.dumps(
                    {"agent_score": self.agent_score, "cpu_score": self.cpu_score}
                ),
            )
            print(f"Computer {self.cpu_score}, Agent {self.agent_score}")
            self.reset()
            return -1

        return None

    def reset(self, total: bool = False) -> None:
        """ Reset Game """

        if total:
            self.cpu_score = 0
            self.agent_score = 0

            self.redis.set(
                "scores",
                json.dumps(
                    {"agent_score": self.agent_score, "cpu_score": self.cpu_score}
                ),
            )

        self.puck.reset()
        self.agent.reset_mallet()
        self.opponent.reset_mallet()

        return None

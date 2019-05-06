""" Air Hockey Game Environment """

import json
from time import time
from typing import Any, Dict, Tuple, Union

import numpy as np
from redis import Redis

from environment.components import Goal, Mallet, Puck
from utils import Action, State, Observation


class AirHockey:

    redis = Redis()

    # Default rewwards
    rewards = {
        "point": 1,
        "loss": -5,
        "hit": 0,
        "miss": 0
    }

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
        self.opponent_score = 0
        self.robot_score = 0
        self._update_score_redis()

        self.ticks_to_friction = 60
        self.ticks_to_ai = 10

        # Define midpoints
        self.table_midpoints = list(map(lambda x: int(x / 2), self.table_size))

        # Define left and right mallet positions
        default_left_position = self.table_midpoints[0] - \
            100, self.table_midpoints[1]
        default_right_position = self.table_midpoints[0] + \
            100, self.table_midpoints[1]

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
        self.robot = Mallet(
            "robot",
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
        self.step_size = 10

        # Set timer for stalling
        self.timer = time()

        # Reward
        self.reward = 0
        self.robot_cumulative_reward = 0
        self.opponent_cumulative_reward = 0

        # If episode is done
        self.done = False

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

    def mallet_ai(self) -> None:
        """ The 'AI' of the computer """

        self.check_stall()

        if self.puck.x < self.opponent.x:
            if self.puck.x < self.opponent.left_lim:
                self.opponent.dx = 1
            else:
                self.opponent.dx = -2

        if self.puck.x > self.opponent.x:
            if self.puck.x > self.opponent.right_lim:
                self.opponent.dx = -1
            else:
                self.opponent.dx = 2

        if self.puck.y < self.opponent.y:
            if self.puck.y < self.opponent.u_lim:
                self.opponent.dy = 1
            else:
                self.opponent.dy = -6

        if self.puck.y > self.opponent.y:
            if self.puck.y <= 360:  # was 250
                self.opponent.dy = 6

            else:
                if self.opponent.y > 200:
                    self.opponent.dy = -2
                else:
                    self.opponent.dy = 0
            # Addresses situation when the puck and the computer are on top of each other.
            # Breaks loop
            if (
                abs(self.puck.y - self.opponent.y) < 40
                and abs(self.puck.x - self.opponent.x) < 40
            ):
                self.puck.dx += 2
                self.puck.dy += 2

        return None

    def _move(self, agent: Mallet, action: Action) -> None:
        """ Move agent's mallet """

        # Update action
        if isinstance(action, tuple) or isinstance(action, list):  # Cartesian Coordinates
            agent.x, agent.y = action[0], action[1]

        # Integers
        if isinstance(action, int) and action == 0:
            agent.y += self.step_size

        if isinstance(action, int) and action == 1:
            agent.y += -self.step_size

        if isinstance(action, int) and action == 2:
            agent.x += -self.step_size

        if isinstance(action, int) and action == 3:
            agent.x += self.step_size

        # Set agent position
        agent.update_mallet()

        # Update agent velocity
        agent.dx = agent.x - agent.last_x
        agent.dy = agent.y - agent.last_y
        agent.update_mallet()

        return None

    def update_state(self, action: Action, agent_name: str = "robot") -> None:
        """ Update state of game """

        # Move mallet
        if agent_name == "robot":
            self._move(self.robot, action)
        elif agent_name == "opponent":
            self._move(self.opponent, action)
        elif agent_name == "human":
            # Update action
            if isinstance(action, tuple) or isinstance(action, list):  # Cartesian Coordinates
                self.opponent.x, self.opponent.y = action[0], action[1]
                self.opponent.update_mallet()
        else:
            raise ValueError("Invalid agent name")

        # Determine puck physics
        if (
            abs(self.robot.x - self.puck.x) <= 50
            and abs(self.robot.y - self.puck.y) <= 50
        ):
            self.puck.dx = -3 * self.puck.dx + self.robot.dx
            self.puck.dy = -3 * self.puck.dy + self.robot.dy

        if (
            abs(self.opponent.x - self.puck.x) <= 50
            and abs(self.opponent.y - self.puck.y) <= 50
        ):
            self.puck.dx = -3 * self.puck.dx + self.opponent.dx
            self.puck.dy = -3 * self.puck.dy + self.opponent.dy

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update_puck()

        # Update puck position
        while self.ticks_to_friction == 0:
            self.puck.friction_on_puck()
            self.ticks_to_friction = 60

        # Update agent position
        self.robot.last_x = self.robot.x
        self.robot.last_y = self.robot.y
        self.robot.update_mallet()

        self.opponent.last_x = self.opponent.x
        self.opponent.last_y = self.opponent.y
        self.opponent.update_mallet()

        # Update score
        self.update_score()

        self.ticks_to_friction -= 1
        self.ticks_to_ai -= 1

        return None

    def _update_score_redis(self) -> None:
        """ Push current score to redis """

        self.redis.set(
            "scores",
            json.dumps({"robot_score": self.robot_score,
                        "opponent_score": self.opponent_score}),
        )

    def update_score(self) -> Union[int, None]:
        """ Get current score """

        # When then agent scores on the computer
        if (
            abs(self.right_goal.centre_y - self.puck.y) <= 50
            and abs(self.right_goal.centre_x - self.puck.x) <= 45
        ):
            self.robot_score += 1
            self.reward = self.rewards["point"] if self.robot_score == 10 else 0
            self.robot_cumulative_reward += self.rewards["point"] if self.robot_score == 10 else 0
            self.opponent_cumulative_reward += self.rewards["loss"] if self.robot_score == 10 else 0

            # Push to redis
            self._update_score_redis()

            print(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.done = True
            self.reset()
            return None

        # When the computer scores on the agent
        if (
            abs(self.left_goal.centre_y - self.puck.y) <= 50
            and abs(self.left_goal.centre_x - self.puck.x) <= 45
        ):
            self.opponent_score += 1
            self.reward = self.rewards["loss"] if self.opponent_score == 10 else 0
            self.robot_cumulative_reward += self.rewards["loss"] if self.opponent_score == 10 else 0
            self.opponent_cumulative_reward += self.rewards["point"] if self.opponent_score == 10 else 0

            # Push to redis
            self._update_score_redis()

            print(f"Agent {self.robot_score}, Computer {self.opponent_score}")
            self.done = True
            self.reset()
            return None

        # # We hit the puck
        # if (
        #     abs(self.robot.x - self.puck.x) <= 35
        #     and abs(self.robot.y - self.puck.y) <= 35
        # ):
        #     self.reward = self.rewards["hit"]
        #     self.cumulative_reward += self.reward
        #     self.done = False
        #     return None

        # self.reward = self.rewards["miss"]
        # self.cumulative_reward += self.reward
        # self.done = False
        return None

    def reset(self, total: bool = False) -> None:
        """ Reset Game """

        if total:
            self.opponent_score = 0
            self.robot_score = 0

            # Push to redis
            self._update_score_redis()

        self.puck.reset()
        self.robot.reset_mallet()
        self.opponent.reset_mallet()

        return None

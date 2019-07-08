""" Air Hockey Game Environment """
import json
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
from redis import Redis

from environment import config
from environment.goal import Goal
from environment.mallet import Mallet
from environment.puck import Puck
from environment.table import Table
from lib.connect import RedisConnection
from lib.types import Action, Observation, State

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AirHockey:
    def __init__(self) -> None:
        """ Initiate an air hockey game """

        # Set up Redis Connection
        self.redis = RedisConnection()

        # Directory to save csv data
        self.stats_dir = ""

        # Create Table
        self.table = Table()

        # Make goals
        self.left_goal = Goal(x=0, y=self.table.midpoints[1])
        self.right_goal = Goal(x=self.table.size[0], y=self.table.midpoints[1])

        # Create puck
        self.puck = Puck(x=self.table.midpoints[0], y=self.table.midpoints[1])

        # Define left and right mallet positions
        mallet_l = self.table.midpoints[0] - 100, self.table.midpoints[1]
        mallet_r = self.table.midpoints[0] + 100, self.table.midpoints[1]

        # Makes Robot Mallet
        self.robot = Mallet(name="robot", x=mallet_l[0], y=mallet_l[1])

        # Makes Computer Mallet
        self.opponent = Mallet(name="opponent", x=mallet_r[0], y=mallet_r[1])

        # Default scores
        self.opponent_score, self.robot_score = 0, 0

        self.mallets = [self.robot, self.opponent]

        # Push to redis
        self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})

        # Drift
        self.ticks_to_friction = 60

        # Define step size of mallet
        self.step_size = 10

    def _move(self, agent: "Mallet", action: "Action") -> None:
        """ Move agent's mallet """

        # Contiunous action
        if isinstance(action, (tuple, list)):  # Cartesian Coordinates
            agent.dx += action[0]
            agent.dy += action[1]

        # Discrete action
        if isinstance(action, int) and action == 0:
            agent.y += self.step_size

        if isinstance(action, int) and action == 1:
            agent.y += -self.step_size

        if isinstance(action, int) and action == 2:
            agent.x += -self.step_size

        if isinstance(action, int) and action == 3:
            agent.x += self.step_size

        # Set agent position
        agent.update()

        # Update agent velocity
        agent.dx = agent.x - agent.last_x
        agent.dy = agent.y - agent.last_y
        agent.update()

        return None

    def update_state(self, action: "Action", agent_name: str = "robot") -> None:
        """ Update state of game """

        # Move mallet
        if agent_name == "robot":
            self._move(self.robot, action)
        elif agent_name == "opponent":
            self._move(self.opponent, action)
        elif agent_name == "human":
            # Update action
            if isinstance(action, (tuple, list)):  # Cartesian Coordinates
                self.opponent.x, self.opponent.y = action[0], action[1]
                self.opponent.update()
        else:
            logger.error("Invalid agent name")
            raise ValueError

        # Check for collisions, do physics magic, update objects
        for mallet in self.mallets:
            self.collision(self.puck, mallet)

        # Update puck position
        self.puck.limit_puck_speed()
        self.puck.update()

        # Update agent and oppponent positions
        self.robot.update()
        self.opponent.update()

        # Implement friction on puck
        # while self.ticks_to_friction == 0:
        #     self.puck.friction_on_puck()
        #     self.ticks_to_friction = 45
        # self.ticks_to_friction -= 1

        # Update Redis
        self.redis.post(
            {
                "components": {
                    "puck": {"location": self.puck.location(), "velocity": self.puck.velocity()},
                    self.robot.name: {"location": self.robot.location(), "velocity": self.robot.velocity()},
                    self.opponent.name: {"location": self.opponent.location(), "velocity": self.opponent.velocity()},
                }
            }
        )

        return None

    # TODO - Add acceleration?
    def get_state(self, agent_name: str = "robot"):
        """ Get current state """

        agent = self.robot if agent_name == "robot" else self.opponent
        state = State(
            agent_location=agent.location(),
            puck_location=self.puck.location(),
            agent_velocity=agent.velocity(),
            puck_velocity=self.puck.velocity(),
        )

        return state

    def update_score(self, score: int) -> Union[int, None]:
        """ Get current score """
        # When then agent scores on the computer
        if score > 0:
            self.robot_score += 1
            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})
            self.redis.publish("score-update")
            logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.reset()
            return None

        # When the computer scores on the agent
        if score < 0:
            self.opponent_score += 1
            # Push to redis
            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})
            self.redis.publish("score-update")
            logger.info(f"Robot {self.robot_score}, Computer {self.opponent_score}")
            self.reset()
            return None
        return None

    def reset(self, total: bool = False) -> None:
        """ Reset Game """

        if total:
            self.opponent_score = 0
            self.robot_score = 0

            # Push to redis
            self.redis.post({"scores": {"robot_score": self.robot_score, "opponent_score": self.opponent_score}})
            self.redis.publish("score-update")
            logger.info("Total Game reset")

        self.puck.reset()
        self.robot.reset()
        self.opponent.reset()

        return None

    @staticmethod
    def collision(puck: "Puck", mallet: "Mallet", correction: bool = True) -> bool:
        """ Collision resolution

        Reference:
            https://www.gamedev.net/forums/topic/488102-circlecircle-collision-response/
            https://gamedevelopment.tutsplus.com/tutorials/how-to-create-a-custom-2d-physics-engine-the-basics-and-impulse-resolution--gamedev-6331
        """
        # separation vector
        d_x = mallet.x - puck.x
        d_y = mallet.y - puck.y
        d = np.array([d_x, d_y])

        #  distance between circle centres, squared
        distance_squared = np.dot(d, d)

        # combined radius squared
        radius = mallet.radius + puck.radius
        radius_squared = radius ** 2

        # No collision
        if distance_squared > radius_squared:
            return False

        # distance between circle centres
        distance = np.sqrt(distance_squared)

        # normal of collision
        ncoll = (d / distance) if distance > 0 else d

        # penetration distance
        dcoll = radius - d

        # Sum of inverse masses
        imass_sum = puck.imass + mallet.imass

        # separation vector
        if correction:
            # For floating point corrections
            percent = config.physics["percent"]  # usually 20% to 80%
            slop = config.physics["slop"]  # usually 0.01 to 0.1
            separation_vector = (np.max(dcoll - slop, 0) / imass_sum) * percent * ncoll
        else:
            separation_vector = (dcoll / imass_sum) * ncoll

        # separate the circles
        puck.x -= separation_vector[0] * puck.imass
        puck.y -= separation_vector[1] * puck.imass
        mallet.x += separation_vector[0] * mallet.imass
        mallet.y += separation_vector[1] * mallet.imass

        # combines velocity
        vcoll_x = mallet.dx - puck.dx
        vcoll_y = mallet.dy - puck.dy
        vcoll = np.array([vcoll_x, vcoll_y])

        # impact speed
        vn = np.dot(vcoll, ncoll)

        # obejcts are moving away. dont reflect velocity
        if vn > 0:
            return True  # we did collide

        # coefficient of restitution in range [0, 1].
        cor = config.physics["restitution"]  # air hockey -> high cor

        # collision impulse
        j = -(1.0 + cor) * (vn / imass_sum)

        # collision impusle vector
        impulse = j * ncoll

        # change momentum of the circles
        puck.dx -= impulse[0] * puck.imass
        puck.dy -= impulse[1] * puck.imass

        mallet.dx += impulse[0] * mallet.imass
        mallet.dy += impulse[1] * mallet.imass

        return True

""" Track rewards """

from datetime import datetime
from typing import List, Tuple

import numpy as np

from environment.components import Goal, Mallet, Puck, Table
from .MemoryBuffer import MemoryBuffer
from .utils import record_data_csv

from .helpers import gaussian
from pytz import timezone


class RewardTracker:

    scores = {"point": 1, "loss": -1}

    def __init__(self, name: str, left_goal: Goal, right_goal: Goal, table: Table):

        self.table = table
        self.left_goal = left_goal
        self.right_goal = right_goal

        self.agent_name = name
        self.std = 50

        self.reward = 0
        self.average_stats = MemoryBuffer(capacity=50000)

        self.puck = None
        self.mallet = None

        self.prev_mallet_to_puck_distance = np.Infinity
        self.prev_puck_to_goal_velocity = 0
        self.prev_puck_to_goal_distance = 0

        self.puck_to_goal_reward = 0.5
        self.mallet_to_puck_reward = 2

    @staticmethod
    def compute_velocity(obj_1: List[int], obj_2: List[int], prev_distance: float) -> List[float]:
        """ Compute velocity """

        new_distance = np.linalg.norm(np.array(obj_1) - np.array(obj_2))
        return new_distance, new_distance - prev_distance

    def compute_wall_reward(self, puck: Puck) -> float:
        """ Compute reward for puck hitting opponent's wall """

        score = 0

        # Hit robot's wall
        if puck << self.table:
            score = -1 * gaussian(puck.y, self.left_goal.y, self.std)

        # Hit opponent's wall
        if puck >> self.table:
            score = gaussian(puck.y, self.right_goal.y, self.std)

        # If we are tracking the reward of the opponent, then flip things around
        score = score if self.agent_name == "robot" else -1 * score
        return score

    def compute_score_reward(self, puck: Puck) -> Tuple[int, int, bool]:
        """ Compute reward for puck hitting opponent's wall """

        score, point = 0, 0

        # Robot Scored
        if puck & self.right_goal and puck | self.right_goal:
            score = self.scores["point"]
            point = 1 if self.agent_name == "robot" else -1

        # Opponent Scored
        if puck & self.left_goal and puck | self.left_goal:
            score = self.scores["loss"]
            point = -1 if self.agent_name == "robot" else 1

        # If we are tracking the reward of the opponent, then flip things around
        score = score if self.agent_name == "robot" else -1 * score
        done = True if abs(score) > 0 else False
        return score, point, done

    def compute_mallet_hit_puck(self, puck: Puck, mallet: Mallet) -> float:
        """ Determine if mallet hit puck """

        if puck & mallet and puck | mallet:
            return 20, True

        return -6, False

    def compute_mallet_to_puck_distance(self, puck: Puck, mallet: Mallet) -> float:
        """ Compute the distance from mallet to puck """

        distance = np.linalg.norm(np.array([puck.x, puck.y]) - np.array([mallet.x, mallet.y]))

        if distance < self.prev_mallet_to_puck_distance:
            self.prev_mallet_to_puck_distance = distance
            return 8  # self.mallet_to_puck_reward

        self.prev_mallet_to_puck_distance = distance
        return -4

    def compute_puck_to_goal_velocity(self, puck: Puck) -> float:
        """ Compute the velocity from puck to goal """

        goal = self.right_goal if self.agent_name == "robot" else self.left_goal

        self.prev_puck_to_goal_distance, velocity = self.compute_velocity(
            [puck.x, puck.y], [goal.x, goal.y], self.prev_puck_to_goal_distance
        )

        if velocity > self.prev_puck_to_goal_velocity:
            self.prev_mallet_to_puck_velocity = velocity
            return self.puck_to_goal_reward

        return -0.3

    def __call__(self, puck: Puck, mallet: Mallet, folder: str = ""):
        """ Compute rewards """

        # rewards, point, done = self.compute_score_reward(puck)
        # rewards += self.compute_wall_reward(puck)
        # rewards += self.compute_puck_to_goal_velocity(puck)
        rewards, done = self.compute_mallet_hit_puck(puck, mallet)
        rewards += self.compute_mallet_to_puck_distance(puck, mallet)
        self.average_stats.append(rewards)
        if done and folder:
            stats = np.array(self.average_stats.retreive())
            mean_average = np.mean(stats)
            record_data_csv(
                folder,
                f"rewards_{self.agent_name}",
                {
                    "created_at": datetime.now(timezone("America/Chicago")),
                    "mean_reward": mean_average,
                    "reward_per_episode": rewards,
                },
            )
            self.average_stats.purge()

        return rewards, point, done

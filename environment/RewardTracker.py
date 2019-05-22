""" Track rewards """

import numpy as np
from typing import List
from environment.components import Goal, Mallet, Puck, Table
from utils import gaussian


class RewardTracker:

    scores = {"point": 1, "loss": -2}

    def __init__(self, name: str, left_goal: Goal, right_goal: Goal, table: Table):

        self.table = table
        self.left_goal = left_goal
        self.right_goal = right_goal

        self.agent_name = name
        self.std = 50

        self.reward = 0
        self.running_average = 0

        self.puck = None
        self.mallet = None

        self.prev_mallet_to_puck_velocity = 0
        self.prev_puck_to_goal_velocity = 0
        self.prev_puck_to_goal_distance = 0

        self.puck_to_goal_reward = 0.2
        self.mallet_to_puck_reward = 0.2

        self.done = False

    @staticmethod
    def compute_velocity(obj_1: List[int], obj_2: List[int], prev_distance: float) -> List[float]:
        """ Compute velocity """

        new_distance = np.linalg.norm(np.array(obj_1) - np.array(obj_2))
        return new_distance, new_distance - prev_distance

    def compute_wall_reward(self, puck: Puck) -> float:
        """ Compute reward for puck hitting opponent's wall """

        score = 0

        # Hit opponent's wall
        if puck & self.right_goal:
            score = gaussian(puck.y, self.right_goal.y, self.std)

        # Hit robot's wall
        if puck & self.left_goal:
            score = -1 * gaussian(puck.y, self.left_goal.y, self.std)

        # If we are tracking the reward of the opponent, then flip things around
        score = score if self.agent_name == "robot" else -1 * score
        return score

    def compute_score_reward(self, puck: Puck) -> float:
        """ Compute reward for puck hitting opponent's wall """

        score = 0

        # Robot Scored
        if puck & self.right_goal and puck | self.right_goal:
            score = self.scores["point"]

        # Opponent Scored
        if puck & self.left_goal and puck | self.left_goal:
            score = self.scores["loss"]

        # If we are tracking the reward of the opponent, then flip things around
        score = score if self.agent_name == "robot" else -1 * score
        self.done = True if abs(score) > 0 else False
        return score

    def compute_mallet_to_puck_velocity(self, puck: Puck, mallet: Mallet) -> float:
        """ Compute the velocity from mallet to puck """

        velocity = np.linalg.norm(np.array([puck.dx, puck.dy]) + np.array([mallet.dx, mallet.dy]))

        if velocity > self.prev_mallet_to_puck_velocity:
            self.prev_mallet_to_puck_velocity = velocity
            return self.mallet_to_puck_reward

        return -0.1

    def compute_puck_to_goal_velocity(self, puck: Puck) -> float:
        """ Compute the velocity from puck to goal """

        goal = self.right_goal if self.agent_name == "robot" else self.left_goal

        self.prev_puck_to_goal_distance, velocity = self.compute_velocity(
            [puck.x, puck.y], [goal.x, goal.y], self.prev_puck_to_goal_distance
        )

        if velocity > self.prev_puck_to_goal_velocity:
            self.prev_mallet_to_puck_velocity = velocity
            return self.puck_to_goal_reward

        return -0.1

    def __call__(self, puck: Puck, mallet: Mallet):
        """ Compute rewards """

        rewards = 0
        rewards += self.compute_wall_reward(puck)
        rewards += self.compute_score_reward(puck)
        rewards += self.compute_mallet_to_puck_velocity(puck, mallet)
        rewards += self.compute_puck_to_goal_velocity(puck)

        return rewards, self.done

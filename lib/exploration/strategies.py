""" Exploration Strategies """
import numpy as np
from typing import Tuple



class SoftmaxPolicy:
    """ Implement softmax policy for multinimial distribution

    Simple Policy
    - takes action according to the pobability distribution
    """

    def __init__(self, action_size: int):
        self.action_size = action_size

    def step(self, probs: np.ndarray, train: bool = True):
        """ Return the selected action """
        return np.random.choice(self.action_size, p=np.nan_to_num(probs))


class EpsilonGreedy:
    """Implement the epsilon greedy policy

    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, action_size: int, **kwargs):
        self.actions = action_size
        self.epsilon = kwargs.get("epsilon", 1)
        self.initial_epsilon = kwargs.get("initial_epsilon", 1)
        self.final_epsilon = kwargs.get("final_epsilon", 0.001)
        self.observe = kwargs.get("observe", 2000)
        self.explore = kwargs.get("explore", 50000)
        self.t = 0

    def decrease(self) -> None:
        """ Decrease epsilon """

        if self.epsilon > self.final_epsilon and self.t % self.observe == 0:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

    def step(self, q_values: np.ndarray) -> int:
        """ Return the selected action """

        # Increment time
        self.t += 1

        # Decrease epsilon
        self.decrease()

        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.actions)
        return np.argmax(q_values)


class BoltzmannQPolicy:
    """Implement the Boltzmann Q Policy

    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """

    def __init__(self, action_size: int, tau: float = 1.0, clip: Tuple[float, float] = (-500.0, 500.0)):
        self.action_size = action_size
        self.tau = tau
        self.clip = clip

    def step(self, q_values: np.ndarray) -> int:
        """ Return the selected action """

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(self.action_size, p=np.nan_to_num(probs))


class MaxBoltzmannQPolicy:
    """
    A combination of the eps-greedy and Boltzman q-policy.

    Wiering, M.: Explorations in Efficient Reinforcement Learning.
    PhD thesis, University of Amsterdam, Amsterdam (1999)

    https://pure.uva.nl/ws/files/3153478/8461_UBA003000033.pdf
    """

    def __init__(self, action_size: int, **kwargs):
        self.action_size = action_size
        self.tau = kwargs.get("tau", 1)
        self.clip = kwargs.get("clip", (-500, 500))
        self.epsilon = kwargs.get("epsilon", 1)
        self.initial_epsilon = kwargs.get("initial_epsilon", 1)
        self.final_epsilon = kwargs.get("final_epsilon", 0.001)
        self.observe = kwargs.get("observe", 2000)
        self.explore = kwargs.get("explore", 50000)
        self.t = 0

    def decrease(self) -> None:
        """ Decrease epsilon """

        if self.epsilon > self.final_epsilon and self.t % self.observe == 0:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore

    def step(self, q_values: np.ndarray) -> int:
        """ Return the selected action
        The selected action follows the BoltzmannQPolicy with probability epsilon
        or return the Greedy Policy with probability (1 - epsilon)
        """

        # Increment time
        self.t += 1

        # Decrease epsilon
        self.decrease()

        if np.random.uniform(0, 1) < self.epsilon:
            exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
            probs = exp_values / np.sum(exp_values)
            return np.random.choice(self.action_size, p=np.nan_to_num(probs))

        return np.argmax(q_values)

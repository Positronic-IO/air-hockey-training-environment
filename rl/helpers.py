""" Helper functions for reinforment learning algorithms """

from typing import Any

import numpy as np
import tensorflow as tf


def huber_loss(y_true, y_pred, clip_delta=1.0) -> float:
    """ Huber loss.
        https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
        https://en.wikipedia.org/wiki/Huber_loss
        """
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)
    return tf.where(cond, squared_loss, linear_loss)


def gaussian(x: int, mu: int, sigma: int) -> float:
    """ Calculate probability of x from some normal distribution """
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


class OrnsteinUhlenbeckProcess:
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """

    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = -self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = (
            self.x0
            + self.theta * (self.mu - self.x0) * self.dt
            + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x0 = x
        return x

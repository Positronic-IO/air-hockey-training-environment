""" Helper functions for reinforment learning algorithms """

from typing import Any

from keras import backend as K
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

    def __call__(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = (
            self.x0
            + self.theta * (self.mu - self.x0) * self.dt
            + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x0 = x
        return x


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        ENTROPY_LOSS = 1e-3

        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -K.mean(
            K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage)
            + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10))
        )

    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        NOISE = 1.0  # Exploration noise

        var = K.square(NOISE)
        denom = K.sqrt(2 * np.pi * var)
        prob_num = K.exp(-K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(-K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num / denom
        old_prob = old_prob_num / denom
        r = prob / (old_prob + 1e-10)

        return -K.mean(
            K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage)
        )

    return loss

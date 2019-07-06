""" Helper functions for reinforment learning algorithms """

from typing import Any

import numpy as np
import tensorflow as tf

from lib.types import State


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
    return tf.keras.backend.exp(-tf.keras.backend.power(x - mu, 2.0) / (2 * tf.keras.backend.power(sigma, 2.0)))


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        ENTROPY_LOSS = 1e-3

        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -tf.keras.backend.mean(
            tf.keras.backend.minimum(
                r * advantage,
                tf.keras.backend.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage,
            )
            + ENTROPY_LOSS * -(prob * tf.keras.backend.log(prob + 1e-10))
        )

    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
        NOISE = 1.0  # Exploration noise

        var = tf.keras.backend.square(NOISE)
        denom = tf.keras.backend.sqrt(2 * np.pi * var)
        prob_num = tf.keras.backend.exp(-tf.keras.backend.square(y_true - y_pred) / (2 * var))
        old_prob_num = tf.keras.backend.exp(-tf.keras.backend.square(y_true - old_prediction) / (2 * var))

        prob = prob_num / denom
        old_prob = old_prob_num / denom
        r = prob / (old_prob + 1e-10)

        return -tf.keras.backend.mean(
            tf.keras.backend.minimum(
                r * advantage,
                tf.keras.backend.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage,
            )
        )

    return loss


def serialize_state(state: "State", dim: int = 2) -> np.ndarray:
    """ Serialize state so that it can be consumed by the model """

    s = np.asarray(state).flatten()
    if dim == 2:
        expanded = np.expand_dims(np.expand_dims(s, axis=0), axis=0)
    else:
        expanded = np.expand_dims(s, axis=0)

    return expanded


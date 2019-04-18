""" Helper functions for reinforment learning algorithms """

import tensorflow as tf
from keras import backend as K


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

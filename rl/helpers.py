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


class TensorBoardLogger:
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir: str):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar variable """

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag: str, values: Any, step: int, bins: int = 100) -> None:
        """Logs the histogram of a list/vector of values."""

        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)

        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

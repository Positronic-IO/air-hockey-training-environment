""" Helper functions for reinforment learning algorithms """

from keras import backend as K


def huber_loss(y_true: float, y_pred: float) -> float:
    """ Compute Huber Loss 
        
        References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
        """
    return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

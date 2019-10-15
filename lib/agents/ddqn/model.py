""" DDQN Neural Network Models """

from typing import Tuple, Union, Dict

import tensorflow as tf
from keras import backend as K
from keras.layers import (
    BatchNormalization,
    Dense,
    Flatten,
    GaussianNoise,
    Input,
    Lambda,
    add,
    Dropout,
    LSTM,
    TimeDistributed,
    Activation,
)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop
from lib.utils.helpers import huber_loss
from lib.utils.noisy_dense import NoisyDense


def config() -> Dict[str, Union[str, int]]:
    return {
        "params": {
            "max_memory": 10000,
            "gamma": 0.95,
            "learning_rate": 0.001,
            "batch_size": 10000,
            "sync_target_interval": 10000,
            "timestep_per_train": 100,
        }
    }


def create(state_size: Tuple[int, int, int], learning_rate: float) -> Model:
    """ DDQN Neural Network """

    model = Sequential()

    model.add(TimeDistributed(NoisyDense(20, kernel_initializer="normal", input_shape=state_size)))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(TimeDistributed(NoisyDense(12, kernel_initializer="normal")))
    model.add(TimeDistributed(Activation("relu")))
    model.add(TimeDistributed(BatchNormalization()))

    model.add(LSTM(40, activation="tanh", return_sequences=True))
    model.add(GaussianNoise(0.7))
    model.add(Dropout(0.7))

    model.add(LSTM(40, activation="tanh", return_sequences=False))
    model.add(GaussianNoise(0.7))
    model.add(Dropout(0.7))

    # model.add(Flatten())

    model.add(NoisyDense(4, kernel_initializer="random_uniform"))
    model.add(Activation("softmax"))

    model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

    return model

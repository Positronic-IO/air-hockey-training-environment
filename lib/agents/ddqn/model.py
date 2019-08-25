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
            "sync_target_interval": 100000,
            "timestep_per_train": 100000,
        }
    }


def create(state_size: Tuple[int, int], learning_rate: float) -> Model:
    """ DDQN Neural Network """

    model = Sequential()

    model.add(Dense(state_size[1], kernel_initializer="random_uniform", input_shape=state_size))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(4, kernel_initializer="random_uniform"))
    model.add(Activation("softmax"))

    model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

    return model

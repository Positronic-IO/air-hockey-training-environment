""" DDQN Neural Network Models """

from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
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
    Activation
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from lib.utils.helpers import huber_loss
from lib.utils.noisy_dense import NoisyDense


def create(state_size: Tuple[int, int], action_size: int, num_atoms: int, learning_rate: float) -> Model:
    """ c51 Neural Net """

    state_input = Input(shape=state_size)

    x = Dense(32, kernel_initializer="normal", activation="relu")(state_input)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    distribution_list = list()
    for _ in range(action_size):
        x = Dense(num_atoms, activation="linear")(x)
        distribution_list.append(x)

    model = Model(state_input, distribution_list)

    model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

    return model

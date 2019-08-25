""" DDQN Neural Network Models """

from typing import Tuple, Union, Dict

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
            "max_memory": 50000,
            "learning_rate": 0.0001,
            "gamma": 0.95,
            "frame_per_action": 4,
            "batch_size": 10000,
            "update_target_freq": 30000,
            "timestep_per_train": 10000,
            "num_atoms": 51,
            "v_max": 10,
            "v_min": -10,
            "iterations_on_save": 10000,
            "epochs": 10,
        }
    }


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

""" DDQN Neural Network Models """

from typing import Tuple, Union, Dict

from keras import backend as K
from keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    GaussianNoise,
    Input,
    Lambda,
    TimeDistributed,
    add,
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
            "learning_rate": 0.00001,
            "gamma": 0.9,
            "epsilon": 1,
            "initial_epsilon": 1,
            "final_epsilon": 0.001,
            "batch_size": 10000,
            "observe": 5000,
            "explore": 50000,
            "frame_per_action": 4,
            "update_target_freq": 30000,
            "timestep_per_train": 10000,
            "iterations_on_save": 10000,
        }
    }


def create(state_size: Tuple[int, int], action_size: int, learning_rate: float) -> Model:
    """ Duelling DDQN Neural Net """

    state_input = Input(shape=state_size)

    x = Dense(32, kernel_initializer="normal", activation="relu")(state_input)
    # x = BatchNormalization()(x)

    x = Flatten()(x)

    # state value tower - V
    state_value = Dense(state_size[1], kernel_initializer="normal", activation="relu")(x)
    state_value = Dense(1, kernel_initializer="random_uniform", activation="linear")(state_value)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(action_size,))(state_value)

    # action advantage tower - A
    action_advantage = Dense(state_size[1], kernel_initializer="normal", activation="relu")(x)
    action_advantage = Dense(action_size, kernel_initializer="random_uniform", activation="linear")(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(
        action_advantage
    )

    # merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(state_input, state_action_value)
    model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

    return model

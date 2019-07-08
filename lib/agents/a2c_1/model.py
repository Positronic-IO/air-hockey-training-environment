""" A2C Neural Network Models """

from typing import Tuple, Union

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
    Activation
)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop

from lib.utils.helpers import huber_loss
from lib.utils.noisy_dense import NoisyDense


def create(
    state_size: Tuple[int, int],
    action_size: int,
    value_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> Tuple[Model, Model]:
    """ A2C Actor and Critic Neural Networks """

    # Actor Network
    actor = Sequential()

    actor.add(Dense(state_size[1], kernel_initializer="normal", input_shape=state_size))
    actor.add(Activation("relu"))
    actor.add(BatchNormalization())

    actor.add(Flatten())

    actor.add(Dense(action_size, kernel_initializer="normal"))
    actor.add(Activation("softmax"))

    actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr=actor_learning_rate))

    # Critic Network
    critic = Sequential()

    critic.add(Dense(state_size[1], kernel_initializer="normal", input_shape=state_size))
    critic.add(Activation("relu"))
    # critic.add(BatchNormalization())

    # critic.add(TimeDistributed(Dense(state_size[1], kernel_initializer="normal", input_shape=state_size)))
    # critic.add(TimeDistributed(Activation("relu")))

    critic.add(Flatten())
    # critic.add(LSTM(state_size[1], activation="tanh", return_sequences=False))
    # critic.add(GaussianNoise(0.6))
    # critic.add(Dropout(0.4))

    critic.add(Dense(value_size, kernel_initializer="random_uniform"))
    critic.add(Activation("linear"))

    critic.compile(loss="mse", optimizer=Adam(lr=critic_learning_rate))

    return actor, critic

""" A2C Neural Network Models """

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
    """ A2C Config """
    return {
        "params": {
            "max_memory": 50000,
            "actor_learning_rate": 0.00001,
            "critic_learning_rate": 0.00001,
            "gamma": 0.95,
            "batch_size": 10000,
            "frame_per_action": 4,
            "timestep_per_train": 10000,
            "iterations_on_save": 10000,
            "epochs": 10,
        }
    }


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

    actor.add(TimeDistributed(NoisyDense(20, kernel_initializer="normal", input_shape=state_size)))
    actor.add(TimeDistributed(Activation("relu")))
    actor.add(TimeDistributed(BatchNormalization()))

    actor.add(TimeDistributed(NoisyDense(12, kernel_initializer="normal")))
    actor.add(TimeDistributed(Activation("relu")))
    actor.add(TimeDistributed(BatchNormalization()))
    
    actor.add(TimeDistributed(NoisyDense(20, kernel_initializer="normal")))
    actor.add(TimeDistributed(Activation("relu")))
    actor.add(TimeDistributed(BatchNormalization()))

    # actor.add(LSTM(40, activation="tanh", return_sequences=True))
    # actor.add(GaussianNoise(0.7))
    # actor.add(Dropout(0.7))

    actor.add(LSTM(40, activation="tanh", return_sequences=False))
    actor.add(GaussianNoise(0.7))
    actor.add(Dropout(0.6))

    actor.add(NoisyDense(action_size, kernel_initializer="normal"))
    actor.add(Activation("softmax"))

    actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr=actor_learning_rate, epsilon=1e-3))

    # Critic Network
    critic = Sequential()

    critic.add(Dense(20, kernel_initializer="normal", input_shape=state_size))
    critic.add(Activation("relu"))
    critic.add(BatchNormalization())

    critic.add(Dense(value_size, kernel_initializer="random_uniform"))
    critic.add(Activation("linear"))

    critic.compile(loss=huber_loss, optimizer=Adam(lr=critic_learning_rate, epsilon=1e-3))

    return actor, critic

""" Reinforcement Learning Neural Network Models """
from typing import Tuple

import tensorflow as tf
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Lambda
from keras.layers.core import Activation, Dense
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop

from rl.helpers import huber_loss


class Networks:
    def __init__(self):
        """ Neural network architectures for Reinforcement Learning algorithms """

    @staticmethod
    def dueling_ddqn(state_size: Tuple[int, int], learning_rate: float) -> Model:
        """ Duelling DDQN Neural Net """

        state_input = Input(shape=state_size)
        x = Dense(12, kernel_initializer="normal", activation="relu")(state_input)
        x = Dense(30, kernel_initializer="normal", activation="relu")(x)
        x = Dense(20, kernel_initializer="normal", activation="relu")(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, kernel_initializer="normal", activation="relu")(x)
        state_value = Dense(4, kernel_initializer="random_uniform")(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(4,))(
            state_value
        )

        # action advantage tower - A
        action_advantage = Dense(256, kernel_initializer="normal", activation="relu")(x)
        action_advantage = Dense(4)(action_advantage)
        action_advantage = Lambda(
            lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(4,)
        )(action_advantage)

        # merge to state-action value function Q
        state_action_value = add([state_value, action_advantage])

        model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

        return model

    @staticmethod
    def dqn(state_size: Tuple[int, int]) -> Model:
        """ Deep Q Neural Network """

        model = Sequential()

        model.add(Dense(12, kernel_initializer="normal", input_shape=state_size))
        model.add(Activation("relu"))

        model.add(Dense(30, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(20, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Flatten())

        model.add(Dense(4, kernel_initializer="random_uniform"))
        model.add(Activation("linear"))

        model.compile(loss=huber_loss, optimizer=RMSprop())

        return model

    @staticmethod
    def ddqn(state_size: Tuple[int, int], learning_rate: float) -> Model:
        """ DDQN Neural Network """

        model = Sequential()

        model.add(Dense(12, kernel_initializer="normal", input_shape=state_size))
        model.add(Activation("relu"))

        model.add(Dense(30, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Dense(20, kernel_initializer="normal"))
        model.add(Activation("relu"))

        model.add(Flatten())

        model.add(Dense(4, kernel_initializer="random_uniform"))
        model.add(Activation("linear"))

        model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

        return model

    @staticmethod
    def c51(
        state_size: Tuple[int, int], action_size: Tuple[int, int], learning_rate: float
    ) -> Model:
        """ c51 Neural Net """

        state_input = Input(shape=state_size)
        x = Dense(12, kernel_initializer="normal", activation="relu")(state_input)
        x = Dense(30, kernel_initializer="normal", activation="relu")(x)
        x = Dense(20, kernel_initializer="normal", activation="relu")(x)
        x = Flatten()(x)

        distribution_list = [
            Dense(51, activation="softmax")(x) for _ in range(action_size)
        ]

        model = Model(input=state_input, output=distribution_list)

        model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

        return model

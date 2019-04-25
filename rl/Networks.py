""" Reinforcement Learning Neural Network Models """
from typing import Tuple

import tensorflow as tf
from keras import backend as K
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input, Lambda, add
from keras.layers.core import Activation, Dense
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop

from rl.helpers import huber_loss


class Networks:
    """ Neural network architectures for Reinforcement Learning algorithms """

    @staticmethod
    def dueling_ddqn(
        state_size: Tuple[int, int, int], action_size: int, learning_rate: float
    ) -> Model:
        """ Duelling DDQN Neural Net """

        state_input = Input(shape=state_size)
        x = Dense(12, kernel_initializer="normal", activation="relu")(state_input)
        x = Dropout(rate=0.3)(x)

        x = Dense(30, kernel_initializer="normal", activation="relu")(x)
        x = Dropout(rate=0.3)(x)

        x = Dense(20, kernel_initializer="normal", activation="relu")(x)
        x = Dropout(rate=0.3)(x)

        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, kernel_initializer="normal", activation="relu")(x)
        state_value = Dropout(rate=0.3)(state_value)

        state_value = Dense(1, kernel_initializer="random_uniform")(state_value)
        state_value = Dropout(rate=0.3)(state_value)

        state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(4,))(
            state_value
        )

        # action advantage tower - A
        action_advantage = Dense(256, kernel_initializer="normal", activation="relu")(x)
        action_advantage = BatchNormalization()(action_advantage)

        action_advantage = Dense(action_size)(action_advantage)
        action_advantage = BatchNormalization()(action_advantage)

        action_advantage = Lambda(
            lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
            output_shape=(action_size,),
        )(action_advantage)

        # merge to state-action value function Q
        state_action_value = add([state_value, action_advantage])

        model = Model(input=state_input, output=state_action_value)
        model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

        return model

    @staticmethod
    def dqn(state_size: Tuple[int, int, int]) -> Model:
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
    def ddqn(state_size: Tuple[int, int, int], learning_rate: float) -> Model:
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
        state_size: Tuple[int, int, int], action_size: int, learning_rate: float
    ) -> Model:
        """ c51 Neural Net """

        state_input = Input(shape=state_size)
        x = Dense(12, kernel_initializer="normal", activation="relu")(state_input)
        x = BatchNormalization()(x)

        x = Dense(30, kernel_initializer="normal", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Dense(20, kernel_initializer="normal", activation="relu")(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)

        distribution_list = list()
        for _ in range(action_size):
            distribution_list.append(Dense(51, activation="softmax")(x))

        model = Model(input=state_input, output=distribution_list)

        model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

        return model

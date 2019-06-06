""" Reinforcement Learning Neural Network Models """

from typing import Tuple

from keras import backend as K
from keras.layers import Dense, Flatten, Input, Lambda, add, BatchNormalization, GaussianNoise
from keras.layers.core import Activation
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, RMSprop

from .helpers import huber_loss


def dueling_ddqn(state_size: Tuple[int, int], action_size: int, learning_rate: float) -> Model:
    """ Duelling DDQN Neural Net """

    state_input = Input(shape=state_size)

    x = Dense(state_size[1], kernel_initializer="glorot_normal", activation="relu")(state_input)
    x = BatchNormalization()(x)

    x = Dense(state_size[1] // 2, kernel_initializer="glorot_normal", activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(state_size[1] // 2, kernel_initializer="glorot_normal", activation="relu")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    # state value tower - V
    state_value = Dense(state_size[1], kernel_initializer="glorot_normal", activation="relu")(x)

    state_value = Dense(1, kernel_initializer="glorot_uniform", activation="linear")(state_value)

    state_value = Lambda(lambda s: K.expand_dims(s[:, 0]), output_shape=(action_size,))(state_value)

    # action advantage tower - A
    action_advantage = Dense(state_size[1], kernel_initializer="glorot_normal", activation="relu")(x)

    action_advantage = Dense(action_size, kernel_initializer="glorot_uniform", activation="linear")(action_advantage)

    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(
        action_advantage
    )

    # merge to state-action value function Q
    state_action_value = add([state_value, action_advantage])

    model = Model(input=state_input, output=state_action_value)
    model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

    return model


def ddqn(state_size: Tuple[int, int], learning_rate: float) -> Model:
    """ DDQN Neural Network """

    model = Sequential()

    model.add(Dense(state_size[1], kernel_initializer="random_uniform", input_shape=state_size))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(state_size[1] // 2, kernel_initializer="random_uniform"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(state_size[1] // 2, kernel_initializer="random_uniform"))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(4, kernel_initializer="random_uniform"))
    model.add(Activation("softmax"))

    model.compile(loss=huber_loss, optimizer=Adam(lr=learning_rate))

    return model


def c51(state_size: Tuple[int, int], action_size: int, learning_rate: float) -> Model:
    """ c51 Neural Net """

    state_input = Input(shape=state_size)

    x = Dense(state_size[1], kernel_initializer="random_uniform", activation="relu")(state_input)
    x = BatchNormalization()(x)

    x = Dense(state_size[1] // 2, kernel_initializer="random_uniform", activation="relu")(x)
    x = BatchNormalization()(x)

    x = Dense(state_size[1] // 2, kernel_initializer="random_uniform", activation="relu")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)

    distribution_list = list()
    for _ in range(action_size):
        x = Dense(state_size[1], activation="softmax")(x)
        distribution_list.append(x)

    model = Model(input=state_input, output=distribution_list)

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=learning_rate))

    return model


def a2c(
    state_size: Tuple[int, int],
    action_size: int,
    value_size: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> Model:
    """ A2C Actor and Critic Neural Networks """

    # Actor Network
    actor = Sequential()

    actor.add(Dense(state_size[1], kernel_initializer="random_uniform", input_shape=state_size))
    actor.add(Activation("relu"))
    actor.add(BatchNormalization())

    actor.add(Dense(state_size[1] // 2, kernel_initializer="random_uniform"))
    actor.add(Activation("relu"))
    actor.add(BatchNormalization())

    actor.add(Dense(state_size[1], kernel_initializer="random_uniform"))
    actor.add(Activation("relu"))
    actor.add(BatchNormalization())

    actor.add(Flatten())

    actor.add(Dense(action_size, kernel_initializer="random_uniform"))
    actor.add(Activation("softmax"))

    actor.compile(loss=huber_loss, optimizer=Adam(lr=actor_learning_rate))

    # Critic Network
    critic = Sequential()

    critic.add(Dense(state_size[1], kernel_initializer="random_uniform", input_shape=state_size))
    critic.add(Activation("relu"))
    critic.add(BatchNormalization())

    critic.add(Dense(state_size[1] // 2, kernel_initializer="random_uniform"))
    critic.add(Activation("relu"))
    critic.add(BatchNormalization())

    critic.add(Dense(state_size[1], kernel_initializer="random_uniform"))
    critic.add(Activation("relu"))
    critic.add(BatchNormalization())

    critic.add(Flatten())

    critic.add(Dense(value_size, kernel_initializer="random_uniform"))
    critic.add(Activation("linear"))

    critic.compile(loss=huber_loss, optimizer=Adam(lr=critic_learning_rate))

    return actor, critic
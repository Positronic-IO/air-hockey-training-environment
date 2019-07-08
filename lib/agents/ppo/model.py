""" DDQN Neural Network Models """

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

from lib.utils.helpers import (
    huber_loss,
    proximal_policy_optimization_loss,
    proximal_policy_optimization_loss_continuous,
)
from lib.utils.noisy_dense import NoisyDense


def create(
    state_size: Tuple[int, int],
    action_size: Union[int, Tuple[int, int]],
    actor_learning_rate: float,
    critic_learning_rate: float,
    continuous: bool = False,
) -> Tuple[Model, Model]:

    # Discrete actor
    def build_actor_discrete(
        state_size: Tuple[int, int], action_size: Union[int, Tuple[int, int]], actor_learning_rate: float
    ) -> Model:
        state_input = Input(shape=state_size)
        advantage_input = Input(shape=(1,))
        old_prediction_input = Input(shape=(action_size,))

        x = Dense(4, kernel_initializer="normal", activation="relu")(state_input)
        x = BatchNormalization()(x)

        x = Dense(6, kernel_initializer="normal", activation="relu")(x)
        # x = BatchNormalization()(x)

        x = Dense(4, kernel_initializer="normal", activation="relu")(x)
        # x = BatchNormalization()(x)

        x = Flatten()(x)

        out_actions = Dense(action_size, kernel_initializer="normal", activation="softmax", name="output")(x)

        model = Model(inputs=[state_input, advantage_input, old_prediction_input], outputs=[out_actions])
        model.compile(
            optimizer=Adam(lr=actor_learning_rate),
            loss=proximal_policy_optimization_loss(advantage=advantage_input, old_prediction=old_prediction_input),
        )
        return model

    # Continuous Actor
    def build_actor_continuous(
        state_size: Tuple[int, int], action_size: Union[int, Tuple[int, int]], actor_learning_rate: float
    ) -> Model:
        state_input = Input(shape=state_size)
        advantage_input = Input(shape=(1,))
        old_prediction_input = Input(shape=(action_size,))

        x = Dense(6, kernel_initializer="normal", activation="tanh")(state_input)
        # x = BatchNormalization()(x)

        x = Dense(6, kernel_initializer="normal", activation="tanh")(x)
        # x = BatchNormalization()(x)

        x = Dense(4, kernel_initializer="normal", activation="tanh")(x)
        # x = BatchNormalization()(x)

        # x = Dense(state_size[1], kernel_initializer="normal", activation="tanh")(x)

        out_actions = Dense(action_size, kernel_initializer="random_uniform", activation="linear", name="output")(
            x
        )

        model = Model([state_input, advantage_input, old_prediction_input], [out_actions])
        model.compile(
            optimizer=Adam(lr=actor_learning_rate),
            loss=proximal_policy_optimization_loss_continuous(
                advantage=advantage_input, old_prediction=old_prediction_input
            ),
        )
        return model

    def build_critic(state_size: Tuple[int, int], critic_learning_rate: float) -> Model:

        # Critic Network
        critic = Sequential()

        critic.add(Dense(6, kernel_initializer="normal", input_shape=state_size))
        critic.add(Activation("tanh"))
        # critic.add(BatchNormalization())

        critic.add(Dense(6, kernel_initializer="normal"))
        critic.add(Activation("tanh"))
        critic.add(BatchNormalization())

        critic.add(Dense(4, kernel_initializer="normal"))
        critic.add(Activation("tanh"))
        # critic.add(BatchNormalization())

        critic.add(Flatten())

        critic.add(Dense(1, kernel_initializer="normal"))
        critic.add(Activation("linear"))

        critic.compile(loss=huber_loss, optimizer=Adam(lr=critic_learning_rate))

        return critic

    actor = object()
    if continuous:
        actor = build_actor_continuous(state_size, action_size, actor_learning_rate)
    else:
        actor = build_actor_discrete(state_size, action_size, actor_learning_rate)

    critic = build_critic(state_size, critic_learning_rate)

    return actor, critic

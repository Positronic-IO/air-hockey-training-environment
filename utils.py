""" Helpful utility functions """

import inspect
import json
import logging
import os
import sys
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union

import numpy as np

# Initiate logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define custom types
Action = Union[int, str, Tuple[int, int]]

State = namedtuple(
    "state", ["robot_location", "puck_location", "robot_velocity", "puck_velocity"]
)

Observation = namedtuple(
    "observation", ["state", "action", "reward", "done", "new_state"]
)


def get_config_strategy(strategy: str) -> Dict[str, Union[str, int]]:
    """ Grab config for different strategies """

    strategies = {
        "q-learner": os.path.join(os.getcwd(), "configs", "q-learner.json"),
        "ddqn": os.path.join(os.getcwd(), "configs", "ddqn.json"),
        "dueling": os.path.join(os.getcwd(), "configs", "dueling.json"),
        "c51": os.path.join(os.getcwd(), "configs", "c51.json"),
        "a2c": os.path.join(os.getcwd(), "configs", "a2c.json"),
    }

    try:
        filename = strategies[strategy]
    except KeyError:
        logger.error("Strategy not defined")
        raise KeyError
    else:
        with open(filename, "r") as f:
            config = json.load(f)
            return config


def gaussian(x: int, mu: int, sigma: int) -> float:
    """ Calculate probability of x from some normal distribution """
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def unique_directory(directory: str) -> str:
    """ Create a unique directory  """

    counter = 0
    while True:
        counter += 1
        path = os.path.join(directory, str(counter))
        if not os.path.exists(path):
            os.makedirs(path)
            return path


def record_model_info(robot: str, opponent: str) -> None:
    """ Record model information """

    from rl import networks

    strategies = {
        "ddqn": networks.ddqn,
        "dueling": networks.dueling_ddqn,
        "c51": networks.c51,
        "a2c": networks.a2c,
    }

    directory = unique_directory(os.path.join(os.getcwd(), "model"))

    try:
        # Deal with robot's models
        robot_path = os.path.join(directory, "robot")
        os.mkdir(robot_path)
        with open(
            os.path.join(robot_path, strategies.get(robot).__name__), "w+"
        ) as file:
            file.write(inspect.getsource(strategies.get(robot)))

        # Deal with opponent's models
        if opponent == "human":
            return robot_path, None

        opponent_path = os.path.join(directory, "opponent")
        os.mkdir(opponent_path)
        with open(
            os.path.join(opponent_path, strategies.get(opponent).__name__), "w+"
        ) as file:
            file.write(inspect.getsource(strategies.get(opponent)))

    except KeyError:
        logger.error("Strategy not defined.")

    # Return base paths of models
    return robot_path, opponent_path

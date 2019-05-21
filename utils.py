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
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        "dqn": os.path.join(os.getcwd(), "configs", "dqn.json"),
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

            if strategy == "a2c":
                if not config["actor"].get("save") or not config["critic"].get("save"):
                    logger.error("Please specify a path to save model.")

            elif strategy != "q-learner" and not config.get("save"):
                logger.error("Please specify a path to save model.")

        return config


def get_model_path(file_path: str) -> str:
    """ Prepare file path for a model to be saved or loaded, create the parent directory if not exists """

    if not os.path.exists(file_path):

        # Split to head and tail
        head, tail = os.path.split(file_path)

        # All weights are stored in the 'models' folder
        if not head:
            return os.path.join("models", tail)

        # Return file with directories made
        os.makedirs(head, exist_ok=True)
        return os.path.join(head, tail)

    return file_path


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
        "dqn": networks.dqn,
        "ddqn": networks.ddqn,
        "dueling": networks.dueling_ddqn,
        "c51": networks.c51,
        "a2c": networks.a2c,
    }

    directory = unique_directory(os.path.join(os.getcwd(), "model"))
    
    try:
        # Deal with robot's models
        path = os.path.join(directory, "robot")
        os.mkdir(path)
        with open(os.path.join(path, strategies.get(robot).__name__), "w+") as file:
            file.write(inspect.getsource(strategies.get(robot)))

        # Deal with opponent's models
        if opponent == "human":
            return None
        
        path = os.path.join(directory, "opponent")
        os.mkdir(path)
        with open(os.path.join(path, strategies.get(opponent).__name__), "w+") as file:
            file.write(inspect.getsource(strategies.get(opponent)))
    
    except KeyError:
        logger.error("Strategy not defined.")

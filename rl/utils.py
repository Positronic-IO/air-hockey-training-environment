""" Helpful utility functions """

import csv
import inspect
import json
import logging
import os
import shutil
import sys
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union

# Initiate logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define custom types
Action = Union[int, str, Tuple[int, int]]

State = namedtuple("state", ["robot_location", "puck_location", "robot_velocity", "puck_velocity"])

Observation = namedtuple("observation", ["state", "action", "reward", "done", "new_state"])


def get_config_strategy(strategy: str) -> Dict[str, Union[str, int]]:
    """ Grab config for different strategies """

    strategies = {
        "q-learner": os.path.join(os.getcwd(), "configs", "q-learner.json"),
        "ddqn": os.path.join(os.getcwd(), "configs", "ddqn.json"),
        "dueling": os.path.join(os.getcwd(), "configs", "dueling.json"),
        "c51": os.path.join(os.getcwd(), "configs", "c51.json"),
        "a2c": os.path.join(os.getcwd(), "configs", "a2c.json"),
        "ppo": os.path.join(os.getcwd(), "configs", "ppo.json"),
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


def unique_directory(directory: str) -> str:
    """ Create a unique directory  """

    counter = 0
    while True:
        counter += 1
        path = os.path.join(directory, str(counter))
        if not os.path.exists(path):
            os.makedirs(path)
            return path, counter


def record_model_info(robot: str, opponent: str) -> None:
    """ Record model information """

    from rl import networks

    strategies = {
        "ddqn": networks.ddqn,
        "dueling": networks.dueling_ddqn,
        "c51": networks.c51,
        "a2c": networks.a2c,
        "ppo": networks.ppo,
    }

    configs = {
        "q-learner": os.path.join(os.getcwd(), "configs", "q-learner.json"),
        "ddqn": os.path.join(os.getcwd(), "configs", "ddqn.json"),
        "dueling": os.path.join(os.getcwd(), "configs", "dueling.json"),
        "c51": os.path.join(os.getcwd(), "configs", "c51.json"),
        "a2c": os.path.join(os.getcwd(), "configs", "a2c.json"),
        "a2c_1": os.path.join(os.getcwd(), "configs", "a2c_1.json"),
        "ppo": os.path.join(os.getcwd(), "configs", "ppo.json"),
    }

    directory, counter = unique_directory(os.path.join(os.getcwd(), "model"))

    try:
        # Deal with robot's models
        robot_path = os.path.join(directory, "robot")
        os.mkdir(robot_path)

        with open(os.path.join(robot_path, strategies.get(robot).__name__), "w+") as file:
            file.write(inspect.getsource(strategies.get(robot)))  # Record model info
            shutil.copy(configs[robot], robot_path)  # Record hyperparameters for model

        # Deal with opponent's models
        if opponent == "human":
            return robot_path, None, counter

        opponent_path = os.path.join(directory, "opponent")
        os.mkdir(opponent_path)

        with open(os.path.join(opponent_path, strategies.get(opponent).__name__), "w+") as file:
            file.write(inspect.getsource(strategies.get(opponent)))  # Record model info
            shutil.copy(configs[opponent], opponent_path)  # Record hyperparameters for model

    except KeyError:
        logger.error("Strategy not defined.")

    from rl.RewardTracker import RewardTracker

    with open(os.path.join(directory, "RewardTracker"), "w+") as file:
        file.write(inspect.getsource(RewardTracker))  # Record reward info

    # Return base paths of models
    return robot_path, opponent_path, counter


def record_data_csv(folder: str, name: str, payload: Any) -> None:
    """ Save data in csv """

    with open(os.path.join("model", folder, name + ".csv"), "a") as file:
        fieldnames = payload.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(payload)

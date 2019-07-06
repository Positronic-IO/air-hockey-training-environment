""" IO helpers """

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

    from lib.agents.a2c import model as a2c_model
    from lib.agents.a2c_1 import model as a2c_1_model
    from lib.agents.c51 import model as c51_model
    from lib.agents.ddqn import model as ddqn_model
    from lib.agents.dueling import model as dueling_model
    from lib.agents.ppo import model as ppo_model

    strategies = {
        "a2c": a2c_1_model,
        "a2c_1": a2c_1_model,
        "c51": c51_model,
        "ddqn": ddqn_model,
        "dueling": dueling_model,
        "ppo": ppo_model,
    }

    directory, counter = unique_directory(os.path.join(os.getcwd(), "model"))

    try:
        # Deal with robot's models
        robot_path = os.path.join(directory, "robot")
        os.mkdir(robot_path)

        with open(os.path.join(robot_path, "model.py"), "w+") as file:
            file.write(inspect.getsource(strategies.get(robot)))  # Record model info
            shutil.copy(
                os.path.join("lib", "agents", robot, "config.py"), robot_path
            )  # Record hyperparameters for model

        if opponent == "human":
            return robot_path, None, counter

        opponent_path = os.path.join(directory, "opponent")
        os.mkdir(opponent_path)

        # Deal with opponent's models
        with open(os.path.join(opponent_path, "model.py"), "w+") as file:
            file.write(inspect.getsource(strategies.get(opponent)))  # Record model info
            shutil.copy(
                os.path.join("lib", "agents", opponent, "config.py"), opponent_path
            )  # Record hyperparameters for model

    except KeyError:
        logger.error("Strategy not defined.")

    with open(os.path.join(directory, "rewards.py"), "w+") as file:
        from lib import rewards

        file.write(inspect.getsource(rewards))  # Record reward info

    # Return base paths of models
    return robot_path, opponent_path, counter


def record_data_csv(folder: str, name: str, payload: Any) -> None:
    """ Save data in csv """

    with open(os.path.join("model", folder, name + ".csv"), "a") as file:
        fieldnames = payload.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(payload)

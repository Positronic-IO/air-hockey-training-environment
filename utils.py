""" Helpful utility functions """

import argparse
import json
import os
import sys
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

# Define custom types
Action = Union[int, str, Tuple[int, int]]

State = namedtuple(
    "state",
    [
        "agent_location",
        "puck_location",
        "puck_prev_location",
        "puck_velocity",
        "opponent_location",
        "opponent_prev_location",
        "opponent_velocity",
    ],
)

Observation = namedtuple(
    "observation", ["state", "action", "reward", "done", "new_state"]
)


#  Load configuration
with open("./config.json", "r") as f:
    config = json.load(f)


def get_config_strategy(name: str) -> Dict[str, Union[str, int]]:
    """ Grab config for different strategies """

    strategies = {
        "dqn": os.path.join(os.getcwd(), "rl", "configs", "dqn.json"),
        "ddqn": os.path.join(os.getcwd(), "rl", "configs", "ddqn.json"),
        "dueling": os.path.join(os.getcwd(), "rl", "configs", "dueling-ddqn.json"),
        "c51": os.path.join(os.getcwd(), "rl", "configs", "c51.json"),
    }

    try:
        with open(strategies[name], "r") as f:
            config = json.load(f)
    except KeyError:
        raise KeyError("Strategy not defined")

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


def write_results(filename: str, data: Dict[str, List[int]]) -> None:
    """ Write data to csv """

    df = pd.DataFrame.from_dict(data)
    if not ".csv" in filename:
        print("Filename to save results does not have the '.csv' extension.")
        sys.exit()

    with open(filename, "a") as f:
        df.to_csv(f, header=False)
    return None

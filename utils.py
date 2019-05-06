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

State = namedtuple("state", ["robot_location", "puck_location", "opponent_location"])

Observation = namedtuple(
    "observation", ["state", "action", "reward", "done", "new_state"]
)


def parse_args() -> Dict[str, Union[str, int]]:

    parser = argparse.ArgumentParser(description="Process stuff for training.")

    parser.add_argument("-r", "--robot", help="Robot strategy")
    parser.add_argument("-o", "--opponent", help="Opponent strategy")
    parser.add_argument(
        "-c", "--capacity", default=5, help="Number of past expierences to store"
    )
    parser.add_argument(
        "-t", "--time", default=3, help="Time per train. Units in hours. (Default to 3 hours)"
    )
    parser.add_argument(
        "--tensorboard",
        help="Tensorbaord log location. If none is specified, then Tensorboard will not be used."
    )
    args = vars(parser.parse_args())
    print(args)

    # Validation
    if not args.get("robot"):
        print("Robot strategy Undefined")
        sys.exit()
    
    if not args.get("opponent"):
        print("Opponent strategy Undefined")
        sys.exit()

    return args


#  Load configuration
def get_config() -> Dict[str, Any]:
    """ Load main config file """

    with open(os.path.join(os.getcwd(), "config.json"), "r") as f:
        config = json.load(f)
        return config


def get_config_strategy(strategy: str) -> Dict[str, Union[str, int]]:
    """ Grab config for different strategies """

    strategies = {
        "q-learner": os.path.join(os.getcwd(), "configs", "q-learner.json"),
        "dqn": os.path.join(os.getcwd(), "configs", "dqn.json"),
        "ddqn": os.path.join(os.getcwd(), "configs", "ddqn.json"),
        "dueling": os.path.join(os.getcwd(), "configs", "dueling.json"),
        "c51": os.path.join(os.getcwd(), "configs", "c51.json"),
    }
    
    try:
        filename = strategies[strategy]
    except KeyError:
        raise KeyError("Strategy not defined")

    with open(filename, "r") as f:
        config = json.load(f)

        if strategy != "q-learner" and not config.get("save"):
            raise KeyError("Please specify a path to save model.")
            
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

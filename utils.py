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

Observation = namedtuple("observation", ["state", "action", "reward", "done", "new_state"])


#  Load configuration
with open("./config.json", "r") as f:
    config = json.load(f)


# def parse_config_agent() -> None:
#     """ Parse arguments for agent settings """

#     print("-" * 35)

#     print("Agent: Robot")
#     print(f"Learning Algorithm: {config["training"]["agent"]['strategy']}")

#     if args.get("load"):
#         print(f"Loading model at: {config["training"]["agent"]['load']}")

#     if args.get("save"):
#         print(f"Saving model at: {config["training"]["agent"]['save']}")

#     if args.get("results"):
#         print(f"Saving results at: {config["training"]["agent"]['results']}")

#     print("-" * 35)

#     return None


def parse_args_gui() -> Dict[str, str]:
    """ Parse arguments for rendering """

    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--agent", default="robot", help="Agent for gameplay")
    ap.add_argument("-l", "--load", help="Load model for robot agent")
    ap.add_argument("-s", "--strategy", help="Playing strategy")
    ap.add_argument(
        "--fps",
        default=-1,
        help="Define FPS of game. A value of -1 allows for the highest possible frame rate. Default to -1",
    )

    args = vars(ap.parse_args())

    print("-" * 35)
    print(f"Agent: {args['agent']}")
    print(f"FPS: {args['fps']}")
    
    if args.get("load") and args.get("strategy"):
        print(f"Strategy: {args['strategy']}")
        print(f"Model name: {args['load']}")
    
    print("-" * 35)

    return args


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

""" Helpful utility functions """

import argparse
import os
import sys
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

# Define custom types
Action = Union[str, Tuple[int, int]]

State = namedtuple("state", ["agent_state", "puck_state", "puck_prev_state"])

Observation = namedtuple("observation", ["state", "action", "reward", "new_state"])


def parse_args() -> Dict[str, str]:
    """ Construct the argument parse and parse the arguments """

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", default="gui", help="Game play")
    ap.add_argument("-a", "--agent", default="human", help="Agent for gameplay")
    ap.add_argument("--fps", default=60, help="Define FPS of game. Defaults to 60 fps. A value of -1 allows for the highest possible frame rate.")
    ap.add_argument("--strategy", default="q-learner", help="Learning strategy")
    ap.add_argument("--load", help="Path to load a model")
    ap.add_argument("--save", help="Path to save a model")
    ap.add_argument("--env", default="normal", help="Define environment")
    ap.add_argument("--results", help="Path to save results as csv")

    args = vars(ap.parse_args())

    # Validations
    if args.get("mode") not in ["gui", "cli"]:
        print("Incorrect game mode.")
        sys.exit()

    if args.get("fps") and args.get("mode") != "gui":
        print(f"Cannot define fps for mode {args.get('mode')}")
        sys.exit()

    if args.get("agent") == "human" and args.get("mode") == "cli":
        print("Human agent only allowed in gui mode.")
        sys.exit()

    if args.get("agent") not in ["human", "robot"]:
        print("Select an allowed agent: human or robot")
        sys.exit()

    if args.get("strategy") not in ["q-learner", "dqn", "ddqn", "dueling-ddqn"]:
        print("Unsupported learning strategy.")
        sys.exit()

    if args.get("load") and not args.get("save"):
        print(
            "Since path is not defined, model will be saved to the same path as it was loaded from."
        )

    return args


def welcome(args: Dict[str, str]) -> None:
    """ Displays welcome to user """

    # Display Welcome
    print(
        """

    #################################################################
    #                   A       IIIII    RRRRRRR                    #
    #                  A A        I      R     R                    #
    #                 A   A       I      RRRRRR                     #
    #                AAAAAAA      I      R    R                     #
    #               A       A   IIIII    R     R                    #
    #                                                               #
    #     H    H     OOO        CCC   K    KK   EEEEEE  YY     YY   #
    #     H    H   OO   OO    CC      K KK      E         YY  YY    #
    #     HHHHHH  O       O  C        K         EEEEEE      YY      #
    #     H    H   OO   OO    CC      K KKK     E           YY      #
    #     H    H     OOO        CCC   K    KK   EEEEEE      YY      #
    #                                                               #
    #                                  By: (Edward) Sum Lok Yu      #
    #                                  Modified By: Tony Hammack    #
    #                                                               #
    #################################################################

    Games go up to 10.
    Move your mouse to control your mallet (the lower one).
    Click "R" to reset the game.

    """
    )

    print("-" * 35)
    if args.get("mode"):
        print(f"Game Mode: {args['mode']}")

    if args.get("fps") and args.get("mode") == "gui":
        print(f"FPS: {args.get('fps')}")

    if args.get("env"):
        print(f"Environment: {args['env']}")

    if args.get("agent") == "human":
        print("Agent: Human")

    if args.get("agent") == "robot":
        print("Agent: Robot")
        print(f"Learning Algorithm: {args['strategy']}")

    if args.get("load"):
        print(f"Loading model at: {args['load']}")

    if args.get("save"):
        print(f"Saving model at: {args['save']}")

    if args.get("results"):
        print(f"Saving results at: {args['results']}")
    print("-" * 35)


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

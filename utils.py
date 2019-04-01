""" Helpful utility functions """

import argparse
import os
from typing import Dict


def parse_args() -> Dict[str, str]:
    """ Construct the argument parse and parse the arguments """

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-a", "--agent", default="human", required=True, help="Agent for gameplay"
    )
    ap.add_argument("--learner", default="q-learner", help="Learner method")
    ap.add_argument("--load", help="Path to load a model")
    ap.add_argument("--save", help="Path to save a model")
    ap.add_argument("--env", default="normal", help="Define environment")

    args = vars(ap.parse_args())
    return args


def welcome(args: dict) -> None:
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
    if args.get("env"):
        print(f"Environment: {args['env']}")

    if args["agent"] == "human":
        print("Agent: Human")

    if args["agent"] == "robot":
        print("Agent: Robot")
        print(f"Learning Algorithm: {args['learner']}")

    if args.get("load"):
        print(f"Loading model at: {args['load']}")

    if args.get("save"):
        print(f"Saving model at: {args['save']}")


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

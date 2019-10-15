""" IO helpers """

import csv
import inspect
import glob
import json
import logging
import os
import shutil
import sys
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union

from lib.utils.exceptions import ProjectNotFoundError, StrategyNotFoundError

# Initiate logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_runid(path: str) -> Tuple[int, str]:
    """ Get runid number for training run """
    listing = glob.glob(f"{path}/*")
    runid = max([0] + [int(x.split("/")[-1].split(".")[0]) for x in listing]) + 1
    path = os.path.join(path, str(runid))
    os.makedirs(path)
    return runid, path


def record_reward(path: str) -> None:
    """ Save reward file """

    with open(os.path.join(path, "rewards.py"), "w+") as file:
        from lib import rewards

        file.write(inspect.getsource(rewards))


def record_model(strategy: str, path: str) -> None:
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

    try:
        with open(os.path.join(path, "model.py"), "w+") as file:
            file.write(inspect.getsource(strategies[strategy]))  # Record model info
    except KeyError:
        logger.error(f"Strategy {strategy} not found.")
        raise StrategyNotFoundError(f"Strategy {strategy} not found.")


def record_data(strategy: str, path: str = "") -> None:
    """ Record data for run """
    # Push current path's run into the environment
    path = path or os.getenv("PROJECT")

    record_model(strategy, path)
    record_reward(path)
    return None


def record_data_csv(name: str, payload: Any) -> None:
    """ Save data in csv """
    path = os.getenv("PROJECT")
    if not path:
        return None

    with open(os.path.join(path, f"{name}.csv"), "a") as file:
        fieldnames = payload.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(payload)

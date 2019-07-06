""" Custom Types """
from collections import namedtuple
from typing import Any, Dict, List, Tuple, Union

Action = Union[int, str, Tuple[int, int]]
State = namedtuple("state", ["agent_location", "puck_location", "agent_velocity", "puck_velocity"])
Observation = namedtuple("observation", ["state", "action", "reward", "done", "new_state"])

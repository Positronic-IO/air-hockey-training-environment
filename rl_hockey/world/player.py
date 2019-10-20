from typing import List, Callable, Any, Dict
from rl_hockey.object import ControlledCircle


class Player:
    """
    Class to store player agents for each world
    """

    __slots__: List[str] = ["obj", "control_func", "control_map", "score", "last_action"]

    def __init__(
        self,
        obj,
        control_func,
        control_map: Dict[int, str],
        score: int = 0,
        last_action: int = -1,
    ):
        self.obj = obj
        self.control_func = control_func
        self.control_map = control_map
        self.score = score
        self.last_action = last_action

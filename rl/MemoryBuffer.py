""" Memory buffer """
import random
from collections import deque
from typing import Any, Deque, List, Union

from utils import Observation, State


class MemoryBuffer:
    """ Hold last n states in memory """

    def __init__(self, capacity: int, default: Any = None):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

        # Set up a buffer with default entries
        if default:
            for _ in range(self.capacity):
                self.append(default)

    def append(self, state: Union[State, Observation]) -> None:
        """ Add states into buffer """

        self.buffer.appendleft(state)

        if len(self.buffer) > self.capacity:
            self.buffer.pop()

        assert len(self.buffer) < self.capacity + 1, "Max memory exceeded"

    def retreive(self) -> Tuple[Union[State, Observation]]:
        """ Retrieve last n states """

        return tuple(self.buffer)

    def sample(self, batch_size: int) -> List[Union[State, Observation]]:
        """ Choose a random sample from memory """

        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """ Return amount of items in buffer """

        return len(self.buffer)

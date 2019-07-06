""" Memory buffer """
import random
from collections import deque
from typing import Any, Deque, List, Tuple, Union

from .types import Observation, State

random.seed(0)


class MemoryBuffer:
    """ Hold last n states in memory """

    def __init__(self, capacity: int, default: Any = None):
        
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

        # Set up a buffer with default entries
        if default:
            for _ in range(self.capacity):
                self.append(default)

    def append(self, state: Union["State", "Observation"]) -> None:
        """ Add states into buffer """

        self.buffer.appendleft(state)

        if len(self.buffer) > self.capacity:
            self.buffer.pop()

        assert len(self.buffer) < self.capacity + 1, "Max memory exceeded"

    def retreive(self, average: bool = False) -> Tuple[Union["State", "Observation"]]:
        """ Retrieve last n states """

        def buffer_average(data):
            return tuple(int(sum(col) / len(data)) for col in zip(*data))

        # Return the average of all the points in the buffer
        if average:
            retval = buffer_average(tuple(self.buffer))
            return (retval,)

        return tuple(self.buffer)

    def sample(self, batch_size: int) -> List[Union["State", "Observation"]]:
        """ Choose a random sample from memory """

        return random.sample(self.buffer, batch_size)

    def purge(self, default: Any = None) -> None:
        """ Purge memory """

        del self.buffer
        self.buffer = deque(maxlen=self.capacity)

        # Set up a buffer with default entries
        if default:
            for _ in range(self.capacity):
                self.append(default)

    def __len__(self) -> int:
        """ Return amount of items in buffer """

        return len(self.buffer)

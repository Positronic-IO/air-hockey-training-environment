""" State buffer """
from collections import deque
from typing import List, Tuple

from utils import State


class States:
    """ Hold last n states in memory """

    def __init__(self, capacity: int, default: List[int] = [0, 0]):
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

        # Set up a buffer with default entries
        for _ in range(self.capacity):
            self.append(default)

    def append(self, state: State) -> None:
        """ Add states into buffer """

        self.buffer.append(state)

        if len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def retreive(self) -> List[State]:
        """ Retrieve last n states """

        return list(self.buffer)

    def __len__(self) -> int:
        """ Return amount of items in buffer """

        return len(self.buffer)

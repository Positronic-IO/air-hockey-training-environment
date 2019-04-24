

from utils import State
from collections import deque

class States:
    """ Hold last n states in memory """

    def __init__(self, capacity: int):
        self.capacity
        self.buffer = deque(maxlen=self.capacity)

    def add(self, state: State) -> None:
        """ Add states into buffer """

        self.buffer.add(state)
        
        if len(self.buffer) > self.capacity:
            self.buffer.popleft()

    def retreive(self) -> List[State]:
        """ Retrieve last n states """

        return list(self.buffer)
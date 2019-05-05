import random
random.seed(0)

from rl.MemoryBuffer import MemoryBuffer


def test_memory_buffer_default():
    """ Test if memory buffer holds default values """

    defaults = ([0, 0], [0, 0], [0, 0])
    buffer = MemoryBuffer(3, [0,0])
    assert buffer.retreive() == defaults

def test_memory_buffer_sample():
    """ Test memory buffer sample """

    output = [[6, 6], [4, 2], [9, 3], [4, 8]]

    capacity = 10
    data = [[random.randint(0, 10), random.randint(0, 10)] for _ in range(capacity)]
    buffer = MemoryBuffer(capacity)

    for item in data:
        buffer.append(item)
    
    assert buffer.sample(4) == output
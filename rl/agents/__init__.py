from tensorflow import set_random_seed

from .A2C import A2C
from .c51 import c51
from .DDQN import DDQN
from .DuelingDDQN import DuelingDDQN
from .PPO import PPO
from .QLearner import QLearner

# Set seed for random number generator
set_random_seed(100)

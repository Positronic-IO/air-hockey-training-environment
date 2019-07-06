""" RL Factory Strategy """
from environment import AirHockey
from lib.agents import A2C, A2C_1, DDQN, PPO, Agent, Dueling, QLearner, c51


class Strategy:
    strategies = {"q-learner": QLearner, "ddqn": DDQN, "dueling": Dueling, "c51": c51, "a2c": A2C, "a2c_1": A2C_1, "ppo": PPO}

    def __init__(self):
        pass

    @classmethod
    def make(self, env: "AirHockey", strategy: str, train: bool = True):

        if strategy == "human":
            return Agent(env)

        return self.strategies.get(strategy)(env, train)

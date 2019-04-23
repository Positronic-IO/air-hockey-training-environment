""" Air Hockey Simulator """
import json
import random
import sys
import time
from typing import Union, Dict

import numpy as np

from environment import AirHockey
from rl import Strategy
from utils import Observation, State, get_model_path, write_results, config

try:
    import sentry_sdk

    # sentry_sdk.init("https://f19813407c4f4c8eb66632f8287cd334@sentry.io/1443046")
except ImportError:
    pass

# If we pass a weights file, load it.
# if hasattr(agent, "load_model") and args.get("load"):
#     file_name = get_model_path(args["load"])
#     agent.load_model(file_name)

# if hasattr(agent, "save_model") and not args.get("save") and not args.get("load"):
#     print("Please specify a path to save model.")
#     sys.exit()


class Train:
    def __init__(self):

        # Set up game environment
        self.env = AirHockey()

        # Set up players
        self.agent = Strategy().make(config["training"]["agent"]["strategy"], self.env)

        # If we pass a weights file, load it.
        if hasattr(self.agent, "load_model") and config["training"]["agent"]["load"]:
            file_name = get_model_path(config["training"]["agent"]["load"])
            self.agent.load_model(file_name)

        if (
            hasattr(self.agent, "save_model")
            and not config["training"]["agent"]["save"]
            and not config["training"]["agent"]["load"]
        ):
            print("Please specify a path to save model.")
            sys.exit()

        self.opponent = Strategy().make(
            config["training"]["opponent"]["strategy"], self.env, agent_name="opponent"
        )
        assert isinstance(
            self.opponent,
            Strategy().strategies[config["training"]["opponent"]["strategy"]],
        )

        # If we pass a weights file, load it.
        if (
            hasattr(self.opponent, "load_model")
            and config["training"]["opponent"]["load"]
        ):
            file_name = get_model_path(config["training"]["opponent"]["load"])
            self.agent.load_model(file_name)

        if (
            hasattr(self.opponent, "save_model")
            and not config["training"]["opponent"]["save"]
            and not config["training"]["opponent"]["load"]
        ):
            print("Please specify a path to save model.")
            sys.exit()

        # Interesting and important constants
        self.epoch = 0
        self.iterations_on_save = 10 ** 4
        self.iterations = 1
        self.new = False

        # We begin..
        self.init, self.init_opponent = True, True

        # Cumulative scores
        self.agent_cumulative_score, self.opponent_cumulative_score = 0, 0

    def stats(self) -> None:
        """ Record training stats """

        results = dict()

        if self.env.agent_cumulative_score > self.agent_cumulative_score:
            results["agent"] = [self.env.agent_cumulative_score]
            self.agent_cumulative_score = self.env.agent_cumulative_score
            self.new = True
        else:
            results["agent"] = [self.agent_cumulative_score]

        if self.env.cpu_cumulative_score > self.opponent_cumulative_score:
            results["opponent"] = [self.env.cpu_cumulative_score]
            self.opponent_cumulative_score = self.env.cpu_cumulative_score
            self.new = True
        else:
            results["opponent"] = [self.opponent_cumulative_score]

        if self.env.agent_score == 10:
            results["agent_win"] = [1]
        else:
            results["agent_win"] = [0]

        if self.env.cpu_score == 10:
            results["cpu_win"] = [1]
        else:
            results["cpu_win"] = [0]

        # results["reward_per_episode"] = [self.env.reward_per_episode]

        if self.new:
            write_results(config["training"]["results"], results)
            self.env.reward_per_episode = 0
            self.new = False

        return None

    def main_player(self) -> None:
        """ Main player """

        # For first move, move in a random direction
        if self.init:
            action = np.random.randint(0, len(self.env.actions) - 1)

            # Update game state
            self.agent.move(action)

            self.init = False
        else:
            # Now, let the model do all the work

            # Current state
            state = State(
                agent_location=self.agent.location(),
                puck_location=self.env.puck.location(),
                puck_prev_location=self.env.puck.prev_location(),
                puck_velocity=self.env.puck.velocity(),
                opponent_location=self.env.opponent.location(),
                opponent_prev_location=self.env.opponent.prev_location(),
                opponent_velocity=self.env.opponent.velocity(),
            )

            # Determine next action
            action = self.agent.get_action(state)

            # Update game state
            self.agent.move(action)

            # New state
            new_state = State(
                agent_location=self.agent.location(),
                puck_location=self.env.puck.location(),
                puck_prev_location=self.env.puck.prev_location(),
                puck_velocity=self.env.puck.velocity(),
                opponent_location=self.env.opponent.location(),
                opponent_prev_location=self.env.opponent.prev_location(),
                opponent_velocity=self.env.opponent.velocity(),
            )

            # Record reward
            self.env.reward_per_episode += self.env.reward

            # Observation of the game at the moment
            observation = Observation(
                state=state,
                action=action,
                reward=self.env.reward,
                done=self.env.done,
                new_state=new_state,
            )

            # Update model
            self.agent.update(observation)

            # Save results to csv
            if config["training"]["results"]:
                self.stats()

        # After so many iterations, save motedel
        if (
            hasattr(self.agent, "save_model")
            and self.iterations % self.iterations_on_save == 0
        ):
            if config["training"]["agent"]["save"]:
                path = get_model_path(config["training"]["agent"]["save"])
                self.agent.save_model(path, self.epoch)
            else:
                self.agent.save_model(epoch=self.epoch)
            self.epoch += 1

        return None

    def opponent_player(self) -> None:
        """ Opponent player """

        if config["training"]["opponent"]["strategy"] == "basic":
            while self.env.ticks_to_ai == 0:
                self.env.mallet_ai()
                self.env.ticks_to_ai = 10
            self.env.opponent.update_mallet()
        else:

            puck = json.loads(self.env.redis.get("puck"))
            agent = json.loads(self.env.redis.get("agent"))
            opponent = json.loads(self.env.redis.get("opponent"))

            # # For first move, move in a random direction
            if self.init_opponent:

                action = np.random.randint(0, len(self.env.actions) - 1)

                # Update game state
                self.opponent.move(action)

                self.init_opponent = False
            else:
                # Now, let the model do all the work

                # Current state
                state = State(
                    agent_location=self.opponent.location(),
                    puck_location=puck["position"],
                    puck_prev_location=puck["prev_position"],
                    puck_velocity=puck["velocity"],
                    opponent_location=agent["position"],
                    opponent_prev_location=agent["prev_position"],
                    opponent_velocity=agent["velocity"],
                )

                # Determine next action
                action = self.opponent.get_action(state)

                # Update game state
                self.opponent.move(action)

                # New state
                new_state = State(
                    agent_location=self.opponent.location(),
                    puck_location=puck["position"],
                    puck_prev_location=puck["prev_position"],
                    puck_velocity=puck["velocity"],
                    opponent_location=agent["position"],
                    opponent_prev_location=agent["prev_position"],
                    opponent_velocity=agent["velocity"],
                )

                # Observation of the game at the moment
                observation = Observation(
                    state=state,
                    action=action,
                    reward=(
                        -1 * self.env.reward
                    ),  # Opposite reward of our agent, only works for current reward settings
                    done=self.env.done,
                    new_state=new_state,
                )

                # Update model
                self.opponent.update(observation)

                self.opponent

                # Update agent velocity
                self.env.opponent.dx = self.env.opponent.x - self.env.opponent.last_x
                self.env.opponent.dy = self.env.opponent.y - self.env.opponent.last_y
                self.env.opponent.update_mallet()

                # # After so many iterations, save motedel
                if (
                    hasattr(self.opponent, "save_model")
                    and self.iterations % self.iterations_on_save == 0
                ):
                    if config["training"]["opponent"]["save"]:
                        path = get_model_path(config["training"]["opponent"]["save"])
                        self.opponent.save_model(path, self.epoch - 1)
                    else:
                        self.opponent.save_model(epoch=self.epoch - 1)

        return None

    def run(self) -> None:
        """ Main guts of training """

        # Game loop
        while True:

            # Our Agent
            self.main_player()

            # Our opponent
            self.opponent_player()

            # Update iterator
            self.iterations += 1

            # Compute scores
            if self.env.cpu_score == 10:
                print(f"Computer {self.env.cpu_score}, agent {self.env.agent_score}")
                print("Computer wins!")
                self.env.reset(total=True)

            if self.env.agent_score == 10:
                print(f"Computer {self.env.cpu_score}, agent {self.env.agent_score}")
                print("Agent wins!")
                self.env.reset(total=True)


if __name__ == "__main__":
    # Run program
    train = Train()
    train.run()

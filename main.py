""" Air Hockey Simulator """
import random
import sys
import time
from typing import Union

import numpy as np

from environment import AirHockey
from environment.test import TestAirHockey
from rl import Strategy
from utils import Observation, State, get_model_path, parse_args_agent, write_results


def main() -> None:
    """ Main guts of game """

    # Parse args
    args = parse_args_agent()

    # Initiate game environment
    env = AirHockey()

    # If user is a robot, set learning style for agent
    agent = Strategy().make(args["strategy"], env)

    # If we pass a weights file, load it.
    if hasattr(agent, "load_model") and args.get("load"):
        file_name = get_model_path(args["load"])
        agent.load_model(file_name)

    if hasattr(agent, "save_model") and not args.get("save") and not args.get("load"):
        print("Please specify a path to save model.")
        sys.exit()

    # Epochs
    epoch = 1

    # Number of iterations between saves
    iterations_on_save = 10 ** 4
    iterations = 1

    # We begin..
    init = True

    # Cumulative scores
    agent_cumulative_score, opponent_cumulative_score = 0, 0

    # Game loop
    while True:

        # Set robot step size
        env.step_size = 10

        # For first move, move in a random direction
        if init:
            action = str(np.random.choice(env.actions))

            # Update game state
            agent.move(action)

            init = False
        else:
            # Now, let the model do all the work

            # Current state
            state = State(
                agent_state=agent.location(),
                puck_state=env.puck.location(),
                puck_prev_state=env.puck.prev_location(),
            )

            # Determine next action
            action = agent.get_action(state)

            # Update game state
            agent.move(action)

            # New state
            new_state = State(
                agent_state=agent.location(),
                puck_state=env.puck.location(),
                puck_prev_state=env.puck.prev_location(),
            )

            # Observation of the game at the moment
            observation = Observation(
                state=state, action=action, reward=env.reward(), new_state=new_state
            )

            # Update model
            if args.get("strategy") in ["ddqn", "dueling-ddqn"]:
                agent.update(observation, iterations)
            else:
                agent.update(observation)

        # After so many iterations, save motedel
        if hasattr(agent, "save_model") and iterations % iterations_on_save == 0:
            if args.get("save"):
                path = get_model_path(args["save"])
                agent.save_model(path, epoch)
            else:
                agent.save_model(epoch=epoch)
            epoch += 1
        iterations += 1

        # Compute scores
        if env.cpu_score == 10 and args.get("env") != "test":
            print(f"Computer {env.cpu_score}, agent {env.agent_score}")
            print("Computer wins!")
            env.reset(total=True)

        if env.agent_score == 10 and args.get("env") != "test":
            print(f"Computer {env.cpu_score}, agent {env.agent_score}")
            print("Agent wins!")
            env.reset(total=True)

        # Patience between calculations
        time.sleep(float(args["wait"]))


if __name__ == "__main__":
    # Run program
    main()
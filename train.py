""" Air Hockey Simulator """
import random
import sys
import time
from typing import Union

import numpy as np

from environment import AirHockey
from rl import Strategy
from utils import Observation, State, get_model_path, parse_args_agent, write_results

try:
    import sentry_sdk
    # sentry_sdk.init("https://f19813407c4f4c8eb66632f8287cd334@sentry.io/1443046")
except ImportError:
    pass

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

        # For first move, move in a random direction
        if init:
            action = np.random.randint(0, len(env.actions) -1)

            # Update game state
            agent.move(action)

            init = False
        else:
            # Now, let the model do all the work

            # Current state
            state = State(
                agent_location=agent.location(),
                puck_location=env.puck.location(),
                puck_prev_location=env.puck.prev_location(),
                puck_velocity=env.puck.velocity(),
                opponent_location=env.opponent.location(),
                opponent_prev_location=env.opponent.prev_location(),
                opponent_velocity=env.opponent.velocity(),
            )

            # Determine next action
            action = agent.get_action(state)

            # Update game state
            agent.move(action)

            # New state
            new_state = State(
                agent_location=agent.location(),
                puck_location=env.puck.location(),
                puck_prev_location=env.puck.prev_location(),
                puck_velocity=env.puck.velocity(),
                opponent_location=env.opponent.location(),
                opponent_prev_location=env.opponent.prev_location(),
                opponent_velocity=env.opponent.velocity(),
            )

            # Record reward
            reward = env.reward
            done = env.done
            # Observation of the game at the moment
            observation = Observation(
                state=state,
                action=action,
                reward=reward,
                done=done,
                new_state=new_state,
            )

            # Update model
            agent.update(observation)

            # Save results to csv
            if args.get("results"):
                results = dict()
                new = False

                if env.agent_cumulative_score > agent_cumulative_score:
                    results["agent"] = [env.agent_cumulative_score]
                    agent_cumulative_score = env.agent_cumulative_score
                    new = True
                else:
                    results["agent"] = [agent_cumulative_score]

                if env.cpu_cumulative_score > opponent_cumulative_score:
                    results["opponent"] = [env.cpu_cumulative_score]
                    opponent_cumulative_score = env.cpu_cumulative_score
                    new = True
                else:
                    results["opponent"] = [opponent_cumulative_score]

                if env.agent_score == 10:
                    results["agent_win"] = [1]
                else:
                    results["agent_win"] = [0]

                if env.cpu_score == 10:
                    results["cpu_win"] = [1]
                else:
                    results["cpu_win"] = [0]

                results["cum_reward"] = [env.cumulative_reward]

                if observation.done:
                    results["done"] = [1]
                else:
                    results["done"] = [0]

                if new:
                    write_results(args["results"], results)

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

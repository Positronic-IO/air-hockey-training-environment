""" Air Hockey Simulator """
import random
import sys

import numpy as np
import pygame

from environment import AirHockey
from environment.test import TestAirHockey
from rl import Strategy
from utils import get_model_path, parse_args, welcome, write_results, State
from typing import Union

# Initialize the game engine
pygame.init()

""" Define Game Constants """

# Define some colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# Offest for drawing the center line of table
middle_line_offset = 4.5

# Define frames per sec
fps = 60


def event_processing(env: Union[AirHockey, TestAirHockey]) -> None:
    """ Pygame event processing """

    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        # User pressed down on a key
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                print("Game reset by user..")
                env.reset(total=True)


def draw_table(env: Union[AirHockey, TestAirHockey]) -> None:
    """ Re-renders table """

    # Make screen
    screen = pygame.display.set_mode(env.table_size)
    screen.fill(blue)

    # Base of rink
    pygame.draw.rect(screen, white, (25, 25, env.rink_size[0], env.rink_size[1]), 0)

    # middle section
    pygame.draw.line(
        screen,
        red,
        [
            env.table_midpoints[0],
            ((env.table_size[1] - env.rink_size[1]) / 2) + middle_line_offset,
        ],
        [
            env.table_midpoints[0],
            env.table_size[1]
            - ((env.table_size[1] - env.rink_size[1]) / 2)
            + middle_line_offset,
        ],
        5,
    )

    # rink frame
    pygame.draw.rect(screen, black, (25, 25, env.rink_size[0], env.rink_size[1]), 5)


def draw_screen(env: Union[AirHockey, TestAirHockey]) -> None:
    """ Create GUI """

    # Set title of game window
    pygame.display.set_caption("Air Hockey")

    # Draw table
    draw_table(env)


def rerender_environment(env: Union[AirHockey, TestAirHockey]) -> None:
    """" Re-render environment """

    # Make screen
    screen = pygame.display.set_mode(env.table_size)

    # Draw table
    draw_table(env)

    # Draw left mallet
    pygame.draw.circle(screen, white, [env.agent.x, env.agent.y], 20, 0)
    pygame.draw.circle(screen, black, [env.agent.x, env.agent.y], 20, 1)
    pygame.draw.circle(screen, black, [env.agent.x, env.agent.y], 5, 0)

    # Draw right mallet
    pygame.draw.circle(screen, white, [env.opponent.x, env.opponent.y], 20, 0)
    pygame.draw.circle(screen, black, [env.opponent.x, env.opponent.y], 20, 1)
    pygame.draw.circle(screen, black, [env.opponent.x, env.opponent.y], 5, 0)

    # Draw left goal
    pygame.draw.rect(
        screen,
        green,
        (env.left_goal.x, env.left_goal.y, env.left_goal.w, env.left_goal.h),
        0,
    )

    # Draw right goal
    pygame.draw.rect(
        screen,
        green,
        (env.right_goal.x, env.right_goal.y, env.right_goal.w, env.right_goal.h),
        0,
    )

    # Draw puck
    pygame.draw.circle(screen, black, [env.puck.x, env.puck.y], env.puck_radius, 0)

    pygame.display.flip()


def main() -> None:
    """ Main guts of game """

    # Parse args
    args = parse_args()

    # Set game clock
    # clock = pygame.time.Clock()

    # Initiate game environment
    if args.get("env") == "test":
        # Testing environment
        env = TestAirHockey()
    else:
        env = AirHockey()

    # If user is a robot, set learning style
    if args["agent"] == "robot":
        agent = Strategy().make(args["strategy"], env)

        # If we pass a weights file, load it.
        if hasattr(agent, "load_model") and args.get("load"):
            file_name = get_model_path(args["load"])
            agent.load_model(file_name)

        if (
            hasattr(agent, "save_model")
            and not args.get("save")
            and not args.get("load")
        ):
            print("Please specify a path to save model.")
            sys.exit()

    # If we are in gui mode, set up air hockey table
    if args.get("mode") == "gui":
        # Make gui
        draw_screen(env)

    # Welcome user
    welcome(args)

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
        if args.get("mode") == "gui":
            event_processing(env)

        # Human agent
        if args["agent"] == "human" and args.get("mode") == "gui":
            # Grab and set user position
            pos = pygame.mouse.get_pos()
            env.update_state(action=pos)

        # Robot
        if args["agent"] == "robot":
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

                # Observe state
                data = env.observe()

                # Current state
                state = State(
                    agent_state=agent.location(),
                    puck_state=env.puck.location(),
                    puck_prev_state=env.puck.prev_location(),
                    # opponent_state=env.opponent.location(),
                    # opponent_prev_state=env.opponent.prev_location(),
                )

                # Determine next action
                action = agent.get_action(state)
                # Update game state
                agent.move(action)

                # DDQN
                if args["strategy"] == "ddqn-agent":
                    next_state = State(
                        agent_state=agent.location(),
                        puck_state=env.puck.location(),
                        puck_prev_state=env.puck.prev_location(),
                        # opponent_state=env.opponent.location(),
                        # opponent_prev_state=env.opponent.prev_location(),
                    )
                    agent.remember(state, action, data["reward"], next_state)
                    agent.update(iterations)

                # DQN
                if args["strategy"] == "dqn-agent":
                    # New state
                    next_state = State(
                        agent_state=agent.location(),
                        puck_state=env.puck.location(),
                        puck_prev_state=env.puck.prev_location(),
                        # opponent_state=env.opponent.location(),
                        # opponent_prev_state=env.opponent.prev_location()
                    )

                    # Update state
                    agent.update(next_state, data["reward"])

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
                        results["agent_win"] = 1
                    else:
                        results["agent_win"] = 0

                    if env.cpu_score == 10:
                        results["cpu_win"] = 1
                    else:
                        results["cpu_win"] = 0

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

        if args.get("mode") == "gui":
            rerender_environment(env)

        # frames per second
        # clock.tick(fps)
    pygame.quit()


if __name__ == "__main__":
    # Run program
    main()

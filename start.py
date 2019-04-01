""" Air Hockey Simulator """
import random
import sys

import numpy as np
import pygame

from environment import AirHockey
from environment.test import TestAirHockey
from learners import LearnerFactory
from utils import get_model_path, parse_args, welcome
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


def draw_table(env: Union[AirHockey, TestAirHockey]) -> None:
    """ Renrers table """

    # base of rink
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


if __name__ == "__main__":
    # -------- Main Program Loop -----------

    # Parse args
    args = parse_args()

    # Initiate game environment
    if args.get("env") == "test":
        # Testing environment
        env = TestAirHockey()
    else:
        env = AirHockey()

    # Who will be playing the game
    agent = args["agent"]

    # If user is a robot, set learning style
    if agent == "robot":
        learner = LearnerFactory().make(args["learner"], env)

        # If we pass a weights file, load it.
        if hasattr(learner, "load_model") and args.get("load"):
            file_name = get_model_path(args["load"])
            learner.load_model(file_name)

        if hasattr(learner, "save_model") and not args.get("save") and not args.get("load"):
            print("Please specify a path to save model.")
            sys.exit()

    # If we are in gui mode, set up air hockey table
    if args.get("mode") == "gui":
        pygame.display.set_caption("Air Hockey")  # Set title of game window

        # Set screen size
        screen = pygame.display.set_mode(env.table_size)

        # Set game clock
        clock = pygame.time.Clock()

        # Draw table
        draw_table(env)

        # Welcome user
        welcome(args)

    # Epochs
    epoch = 1

    # Number of iterations between saves
    iterations_on_save = 10 ** 4
    iterations = 1

    # We begin..
    init = True

    # Game loop
    while True:
        # Event processing
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            # User pressed down on a key
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    env.left_mallet.dx = -7
                if event.key == pygame.K_d:
                    env.left_mallet.dx = 7
                if event.key == pygame.K_w:
                    env.left_mallet.dy = -7
                if event.key == pygame.K_s:
                    env.left_mallet.dy = 7

                if event.key == pygame.K_r:
                    print("Game reset by user..")
                    env.reset(total=True)

            # User let up on a key
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    env.left_mallet.dx = 0
                if event.key == pygame.K_d:
                    env.left_mallet.dx = 0
                if event.key == pygame.K_w:
                    env.left_mallet.dy = 0
                if event.key == pygame.K_s:
                    env.left_mallet.dy = 0

        # Human agent
        if agent == "human" and args.get("mode") == "gui":
            # Grab and set user position
            pos = pygame.mouse.get_pos()
            env.update_state(action=pos)

        # Robot
        if agent == "robot":
            # Set robot step size
            env.step_size = 10
            
            # For first move, move in a random direction
            if init:
                action = str(np.random.choice(env.actions))

                # Update game state
                learner.move(action)

                init = False
            else:
                # Now, let the model do all the work

                # Observe state
                data = env.observe()

                # Current state
                state = (data["agent"], data["puck"])

                # Determine next action
                action = learner.get_action(state)

                # Update game state
                learner.move(action)

                # DDQN
                if args["learner"] == "ddqn-learner":
                    next_state = (learner.location(), (env.puck.x, env.puck.y))
                    learner.remember(state, action, data["reward"], next_state)
                    learner.update()

                # DQN
                if args["learner"] == "dqn-learner":
                    # New state
                    next_state = (data["agent"], data["puck"])
                    # Update state
                    learner.update(next_state, data["reward"])

            # After so many iterations, save model
            if hasattr(learner, "save_model") and iterations % iterations_on_save == 0:
                if args.get("save"):
                    path = get_model_path(args["save"])
                    learner.save_model(path, epoch)
                else:
                    learner.save_model(epoch=epoch)
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
            # Re-render environment
            screen.fill(blue)
            draw_table(env)

            # Draw left mallet
            pygame.draw.circle(
                screen, white, [env.left_mallet.x, env.left_mallet.y], 20, 0
            )
            pygame.draw.circle(
                screen, black, [env.left_mallet.x, env.left_mallet.y], 20, 1
            )
            pygame.draw.circle(
                screen, black, [env.left_mallet.x, env.left_mallet.y], 5, 0
            )

            # Draw right mallet
            pygame.draw.circle(
                screen, white, [env.right_mallet.x, env.right_mallet.y], 20, 0
            )
            pygame.draw.circle(
                screen, black, [env.right_mallet.x, env.right_mallet.y], 20, 1
            )
            pygame.draw.circle(
                screen, black, [env.right_mallet.x, env.right_mallet.y], 5, 0
            )

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
                (
                    env.right_goal.x,
                    env.right_goal.y,
                    env.right_goal.w,
                    env.right_goal.h,
                ),
                0,
            )

            # Draw puck
            pygame.draw.circle(
                screen, black, [env.puck.x, env.puck.y], env.puck_radius, 0
            )

            pygame.display.flip()

            # frames per second
            clock.tick(fps)

    pygame.quit()

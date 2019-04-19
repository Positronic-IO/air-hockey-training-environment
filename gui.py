""" Air Hockey Simulator """
import json
from typing import Union

import pygame
from keras.models import load_model
from redis import Redis

from environment import AirHockey
from rl import Strategy
from utils import parse_args_gui, State, get_model_path

# Set up redis
redis = Redis()

# Initialize the game engine
pygame.init()

# Define some colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# Offest for drawing the center line of table
middle_line_offset = 4.5


def draw_table(env: AirHockey) -> None:
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


def draw_screen(env: AirHockey) -> None:
    """ Create GUI """

    # Set title of game window
    pygame.display.set_caption("Air Hockey")

    # Draw table
    draw_table(env)


def rerender_environment(env: AirHockey) -> None:
    """" Re-render environment """

    puck = json.loads(redis.get("puck"))
    agent = json.loads(redis.get("agent"))
    opponent = json.loads(redis.get("opponent"))

    # Make screen
    screen = pygame.display.set_mode(env.table_size)

    # Draw table
    draw_table(env)

    # Draw left mallet
    pygame.draw.circle(screen, white, agent["position"], 20, 0)
    pygame.draw.circle(screen, black, agent["position"], 20, 1)
    pygame.draw.circle(screen, black, agent["position"], 5, 0)

    # Draw right mallet
    pygame.draw.circle(screen, white, opponent["position"], 20, 0)
    pygame.draw.circle(screen, black, opponent["position"], 20, 1)
    pygame.draw.circle(screen, black, opponent["position"], 5, 0)

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
    pygame.draw.circle(screen, black, puck["position"], env.puck_radius, 0)

    pygame.display.flip()


def main() -> None:
    """ Main guts of game """

    # Parse args
    args = parse_args_gui()

    # Set game clock
    clock = pygame.time.Clock()
    fps = float(args.get("fps", -1))

    # Initiate game environment
    env = AirHockey()

    # Make gui
    draw_screen(env)

    if args["agent"] == "robot" and args.get("load"):
        # If user is a robot, set learning style for agent
        agent = Strategy().make(args["strategy"], env)

        # If we pass a weights file, load it.
        if hasattr(agent, "load_model") and args.get("load"):
            file_name = get_model_path(args["load"])
            agent.load_model(file_name)

    # Game loop
    while True:

        # Human agent
        if args["agent"] == "human":
            # Grab and set user position
            pos = pygame.mouse.get_pos()
            env.update_state(action=pos)

        if args["agent"] == "robot" and args.get("load"):
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

        scores = json.loads(redis.get("scores"))

        # Compute scores
        if scores["cpu_score"] == 10:
            print(f"Computer {scores['cpu_score']}, agent {scores['agent_score']}")
            print("Computer wins!")
            if args["agent"] == "human":
                env.reset(total=True)

        if scores["agent_score"] == 10:
            print(f"Computer {scores['cpu_score']}, agent {scores['agent_score']}")
            print("Agent wins!")

            if args["agent"] == "human":
                env.reset(total=True)

        rerender_environment(env)

        # frames per second
        if fps > -1:
            clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    # Run program
    main()

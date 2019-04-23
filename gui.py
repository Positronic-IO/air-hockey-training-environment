""" Air Hockey Simulator """
import json
from typing import Union

import pygame
import numpy as np
from keras.models import load_model
from redis import Redis

from environment import AirHockey
from rl import Strategy
from utils import parse_args_gui, State, get_model_path, config

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


def main_player(env, player, init) -> None:
    """ Main player """

    # For first move, move in a random direction
    if init:
        action = np.random.randint(0, len(env.actions) - 1)

        # Update game state
        player.move(action)

        init = False
    else:
        # Now, let the model do all the work

        # Current state
        state = State(
            agent_location=player.location(),
            puck_location=env.puck.location(),
            puck_prev_location=env.puck.prev_location(),
            puck_velocity=env.puck.velocity(),
            opponent_location=env.opponent.location(),
            opponent_prev_location=env.opponent.prev_location(),
            opponent_velocity=env.opponent.velocity(),
        )

        # Determine next action
        action = player.get_action(state)

        # Update game state
        player.move(action)

    return init


def opponent_player(env, player, init_opponent) -> None:
    """ Opponent player """

    puck = json.loads(env.redis.get("puck"))
    agent = json.loads(env.redis.get("agent"))

    # # For first move, move in a random direction
    if init_opponent:

        action = np.random.randint(0, len(env.actions) - 1)

        # Update game state
        player.move(action)

        init_opponent = False
    else:
        # Now, let the model do all the work

        # Current state
        state = State(
            agent_location=player.location(),
            puck_location=puck["position"],
            puck_prev_location=puck["prev_position"],
            puck_velocity=puck["velocity"],
            opponent_location=agent["position"],
            opponent_prev_location=agent["prev_position"],
            opponent_velocity=agent["velocity"],
        )

        # Determine next action
        action = player.get_action(state)

        # Update game state
        player.move(action)

        # Update agent velocity
        env.opponent.dx = env.opponent.x - env.opponent.last_x
        env.opponent.dy = env.opponent.y - env.opponent.last_y
        env.opponent.update_mallet()

    return init_opponent


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

    init, init_opponent = True, True

    if args["agent"] == "robot" and not config["observe"]:
        # If user is a robot, set learning style for agent
        agent = Strategy().make(
            config["training"]["agent"]["strategy"], env, agent_name="main"
        )

        # If we pass a weights file, load it.
        if hasattr(agent, "load_model") and config["training"]["agent"]["strategy"]:
            file_name = get_model_path(config["training"]["agent"]["load"])
            agent.load_model(file_name)

        opponent = Strategy().make(
            config["training"]["opponent"]["strategy"], env, agent_name="opponent"
        )

        # If we pass a weights file, load it.
        if hasattr(opponent, "load_model") and config["training"]["opponent"]["load"]:
            file_name = get_model_path(config["training"]["opponent"]["load"])
            opponent.load_model(file_name)

    # Game loop
    while True:

        # Human agent
        if args["agent"] == "human":
            # Grab and set user position
            pos = pygame.mouse.get_pos()
            env.update_state(action=pos)

        if args["agent"] == "robot" and not config["observe"]:

            init = main_player(env, agent, init)
            init_opponent = opponent_player(env, opponent, init_opponent)

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

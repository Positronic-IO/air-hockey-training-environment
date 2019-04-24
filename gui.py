""" Air Hockey Simulator Gui """
import json

import numpy as np
import pygame
from keras.models import load_model
from redis import Redis

from environment import AirHockey
from rl import Strategy
from utils import State, config, get_model_path, parse_args_gui

# Initialize the game engine
pygame.init()


class AirHockeyGui:

    # Set up redis
    redis = Redis()

    # Define some colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    # Offest for drawing the center line of table
    middle_line_offset = 4.5

    def __init__(self):

        # Parse args
        self.args = parse_args_gui()

        # Initiate game environment
        self.env = AirHockey()

        # Make gui
        self.draw_screen()

        self.init, self.init_opponent = True, True

        if self.args["agent"] == "robot" and not config["observe"]:
            # If user is a robot, set learning style for agent
            self.main_agent = Strategy().make(
                config["live"]["agent"]["strategy"], self.env, agent_name="main"
            )

            # If we pass a weights file, load it.
            if (
                hasattr(main_agent, "load_model")
                and config["live"]["agent"]["strategy"]
            ):
                self.main_agent.load_path = get_model_path(
                    config["live"]["agent"]["load"]
                )
                self.main_agent.load_model()

            self.opponent_agent = Strategy().make(
                config["live"]["opponent"]["strategy"], self.env, agent_name="opponent"
            )

            # If we pass a weights file, load it.
            if (
                hasattr(self.opponent_agent, "load_model")
                and config["live"]["opponent"]["load"]
            ):
                self.opponent_agent.load_path = get_model_path(
                    config["live"]["opponent"]["load"]
                )
                self.opponent_agent.load_model()

    def draw_table(self) -> None:
        """ Re-renders table """

        # Make screen
        screen = pygame.display.set_mode(self.env.table_size)
        screen.fill(self.blue)

        # Base of rink
        pygame.draw.rect(
            screen,
            self.white,
            (25, 25, self.env.rink_size[0], self.env.rink_size[1]),
            0,
        )

        # middle section
        pygame.draw.line(
            screen,
            self.red,
            [
                self.env.table_midpoints[0],
                ((self.env.table_size[1] - self.env.rink_size[1]) / 2)
                + self.middle_line_offset,
            ],
            [
                self.env.table_midpoints[0],
                self.env.table_size[1]
                - ((self.env.table_size[1] - self.env.rink_size[1]) / 2)
                + self.middle_line_offset,
            ],
            5,
        )

        # rink frame
        pygame.draw.rect(
            screen,
            self.black,
            (25, 25, self.env.rink_size[0], self.env.rink_size[1]),
            5,
        )

    def draw_screen(self) -> None:
        """ Create GUI """

        # Set title of game window
        pygame.display.set_caption("Air Hockey")

        # Draw table
        self.draw_table()

    def rerender_environment(self) -> None:
        """" Re-render environment """

        puck = json.loads(self.redis.get("puck"))
        agent = json.loads(self.redis.get("agent"))
        opponent = json.loads(self.redis.get("opponent"))

        # Make screen
        screen = pygame.display.set_mode(self.env.table_size)

        # Draw table
        self.draw_table()

        # Draw left mallet
        pygame.draw.circle(screen, self.white, agent["position"], 20, 0)
        pygame.draw.circle(screen, self.black, agent["position"], 20, 1)
        pygame.draw.circle(screen, self.black, agent["position"], 5, 0)

        # Draw right mallet
        pygame.draw.circle(screen, self.white, opponent["position"], 20, 0)
        pygame.draw.circle(screen, self.black, opponent["position"], 20, 1)
        pygame.draw.circle(screen, self.black, opponent["position"], 5, 0)

        # Draw left goal
        pygame.draw.rect(
            screen,
            self.green,
            (
                self.env.left_goal.x,
                self.env.left_goal.y,
                self.env.left_goal.w,
                self.env.left_goal.h,
            ),
            0,
        )

        # Draw right goal
        pygame.draw.rect(
            screen,
            self.green,
            (
                self.env.right_goal.x,
                self.env.right_goal.y,
                self.env.right_goal.w,
                self.env.right_goal.h,
            ),
            0,
        )

        # Draw puck
        pygame.draw.circle(
            screen, self.black, puck["position"], self.env.puck_radius, 0
        )
        pygame.display.flip()

    def main_player_move(self) -> None:
        """ Main player """

        puck = json.loads(self.redis.get("puck"))
        opponent = json.loads(self.redis.get("opponent"))

        # For first move, move in a random direction
        if self.init:
            action = np.random.randint(0, len(self.env.actions) - 1)

            # Update game state
            self.main_agent.move(action)

            self.init = False
        else:
            # Now, let the model do all the work

            # Current state
            state = State(
                agent_location=self.main_agent.location(),
                puck_location=puck["position"],
                puck_prev_location=puck["prev_position"],
                puck_velocity=puck["velocity"],
                opponent_location=opponent["position"],
                opponent_prev_location=opponent["prev_position"],
                opponent_velocity=opponent["velocity"],
            )

            # Determine next action
            action = self.main_agent.get_action(state)

            # Update game state
            self.main_agent.move(action)

        return None

    def opponent_player_move(self) -> None:
        """ Opponent player """

        puck = json.loads(self.env.redis.get("puck"))
        agent = json.loads(self.env.redis.get("agent"))

        # # For first move, move in a random direction
        if self.init_opponent:

            action = np.random.randint(0, len(self.env.actions) - 1)

            # Update game state
            self.opponent_agent.move(action)

            self.init_opponent = False
        else:
            # Now, let the model do all the work

            # Current state
            state = State(
                agent_location=self.opponent_agent.location(),
                puck_location=puck["position"],
                puck_prev_location=puck["prev_position"],
                puck_velocity=puck["velocity"],
                opponent_location=agent["position"],
                opponent_prev_location=agent["prev_position"],
                opponent_velocity=agent["velocity"],
            )

            # Determine next action
            action = self.opponent_agent.get_action(state)

            # Update game state
            self.opponent_agent.move(action)

            # Update agent velocity
            self.env.opponent.dx = self.env.opponent.x - self.env.opponent.last_x
            self.env.opponent.dy = self.env.opponent.y - self.env.opponent.last_y
            self.env.opponent.update_mallet()

        return None

    def run(self) -> None:
        """ Main guts of game """

        # Set game clock
        clock = pygame.time.Clock()
        fps = int(self.args.get("fps", -1))

        # Game loop
        while True:

            # Human agent
            if self.args["agent"] == "human":
                # Grab and set user position
                pos = pygame.mouse.get_pos()
                self.env.update_state(action=pos)

            if self.args["agent"] == "robot" and not config["observe"]:

                self.main_player_move()
                self.opponent_player_move()

            scores = json.loads(self.redis.get("scores"))

            # Compute scores
            if scores["cpu_score"] == 10:
                print(f"Computer {scores['cpu_score']}, agent {scores['agent_score']}")
                print("Computer wins!")

                if self.args["agent"] == "human":
                    self.env.reset(total=True)

            if scores["agent_score"] == 10:
                print(f"Computer {scores['cpu_score']}, agent {scores['agent_score']}")
                print("Agent wins!")

                if self.args["agent"] == "human":
                    self.env.reset(total=True)

            self.rerender_environment()

            # frames per second
            if fps > -1:
                clock.tick(fps)

        pygame.quit()


if __name__ == "__main__":
    # Run program
    gui = AirHockeyGui()
    gui.run()

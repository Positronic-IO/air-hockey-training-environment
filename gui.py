""" Air Hockey Simulator Gui """
import json

import numpy as np
import pygame
from keras.models import load_model
from redis import Redis

from environment import AirHockey
from rl import MemoryBuffer, Strategy
from utils import State, get_config, get_model_path

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

        # Load config
        self.config = get_config()

        # Initiate game environment
        self.env = AirHockey()

        # Make gui
        self.draw_screen()

        self.init, self.init_opponent = True, True

        if self.config["robot"] and not self.config["observe"]:
            # If user is a robot, set learning style for agent
            self.main_agent = Strategy().make(
                self.config["live"]["agent"]["strategy"], self.env, agent_name="main"
            )

            # If we pass a weights file, load it.
            if (
                hasattr(self.main_agent, "load_model")
                and self.config["live"]["agent"]["strategy"]
            ):
                self.main_agent.load_path = get_model_path(
                    self.config["live"]["agent"]["load"]
                )
                self.main_agent.load_model()

            self.opponent_agent = Strategy().make(
                self.config["live"]["opponent"]["strategy"],
                self.env,
                agent_name="opponent",
            )

            # If we pass a weights file, load it.
            if (
                hasattr(self.opponent_agent, "load_model")
                and self.config["live"]["opponent"]["load"]
            ):
                self.opponent_agent.load_path = get_model_path(
                    self.config["live"]["opponent"]["load"]
                )
                self.opponent_agent.load_model()

        # Set up buffers for agent position, puck position, opponent position
        self.agent_location_buffer = MemoryBuffer(self.config["capacity"], [0, 0])
        self.puck_location_buffer = MemoryBuffer(self.config["capacity"], [0, 0])
        self.opponent_location_buffer = MemoryBuffer(self.config["capacity"], [0, 0])

        # Update buffers
        self._update_buffers()

    def _update_buffers(self) -> None:
        """ Update redis buffers """

        self.puck_location = json.loads(self.env.redis.get("puck"))["location"]
        self.agent_location = json.loads(self.env.redis.get("agent_mallet"))["location"]
        self.opponent_location = json.loads(self.env.redis.get("opponent_mallet"))[
            "location"
        ]

        self.agent_location_buffer.append(self.agent_location)
        self.puck_location_buffer.append(self.puck_location)
        self.opponent_location_buffer.append(self.opponent_location)

        return None

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

        self._update_buffers()

        # Make screen
        screen = pygame.display.set_mode(self.env.table_size)

        # Draw table
        self.draw_table()

        # Draw left mallet
        pygame.draw.circle(screen, self.white, self.agent_location, 20, 0)
        pygame.draw.circle(screen, self.black, self.agent_location, 20, 1)
        pygame.draw.circle(screen, self.black, self.agent_location, 5, 0)

        # Draw right mallet
        pygame.draw.circle(screen, self.white, self.opponent_location, 20, 0)
        pygame.draw.circle(screen, self.black, self.opponent_location, 20, 1)
        pygame.draw.circle(screen, self.black, self.opponent_location, 5, 0)

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
            screen, self.black, self.puck_location, self.env.puck_radius, 0
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
        fps = self.config["fps"]

        # Game loop
        while True:

            # Human agent
            if not self.config["robot"]:
                # Grab and set user position
                pos = pygame.mouse.get_pos()
                self.env.update_state(action=pos)

            if self.config["robot"] and not self.config["observe"]:

                self.main_player_move()
                self.opponent_player_move()

            scores = json.loads(self.redis.get("scores"))

            # Compute scores
            if scores["cpu_score"] == 10:
                print(f"Computer {scores['cpu_score']}, agent {scores['agent_score']}")
                print("Computer wins!")

            if scores["agent_score"] == 10:
                print(f"Computer {scores['cpu_score']}, agent {scores['agent_score']}")
                print("Agent wins!")

            if not self.config["robot"]:
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

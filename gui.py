""" Air Hockey Simulator Gui """
import argparse
from typing import Dict

import pygame

from connect import RedisConnection
from environment import AirHockey

# Initialize the game engine
pygame.init()


class AirHockeyGui:

    # Set up redis
    redis = RedisConnection()

    # Define some colors
    black = (0, 0, 0)
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    blue = (0, 0, 255)

    # Offest for drawing the center line of table
    middle_line_offset = 4.5

    def __init__(self, args: Dict[str, int]):

        # Set frames per second
        self.fps = args["fps"]

        # Initiate game environment
        self.env = AirHockey()

        # Make gui
        self.draw_screen()

        # Update locations
        self.update_locations()

    def update_locations(self) -> None:
        """ Update locations of puck, robot, and the opponent"""

        data = self.redis.get()
        self.puck_location = data["puck"]["location"]
        self.robot_location = data["robot"]["location"]
        self.opponent_location = data["opponent"]["location"]
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

        self.update_locations()

        # Make screen
        screen = pygame.display.set_mode(self.env.table_size)

        # Draw table
        self.draw_table()

        # Draw left mallet
        pygame.draw.circle(screen, self.white, self.robot_location, 20, 0)
        pygame.draw.circle(screen, self.black, self.robot_location, 20, 1)
        pygame.draw.circle(screen, self.black, self.robot_location, 5, 0)

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

    def run(self) -> None:
        """ Main guts of game """

        # Set game clock
        clock = pygame.time.Clock()

        # Game loop
        while True:

            self.rerender_environment()

            # frames per second
            clock.tick(self.fps)

        pygame.quit()


if __name__ == "__main__":
    """ Start Gui """
    parser = argparse.ArgumentParser(description="Pygame gui")

    parser.add_argument(
        "--fps", default=60, help="Frames per second. Default is 60 fps"
    )
    args = vars(parser.parse_args())

    # Initialize gui program
    gui = AirHockeyGui(args)
    gui.run()

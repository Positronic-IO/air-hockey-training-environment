from .world import World
from .player import Player
from rl_hockey.object.objects import StaticObject, DynamicObject, ControlledCircle
from rl_hockey.object.shapes import LineShape, CircleShape

import numpy as np


class Hockey1v1Heuristic(World):
    """
    World based on air hockey.

    This is a 1v1 version. The left agent uses the model-in-training, while the right agent
    uses a simple heuristic controller that chases the puck and tries to stay behind it.
    """

    def __init__(self):

        super(Hockey1v1Heuristic, self).__init__(world_size=[2000, 1000])

        self.control_map = [
            {
                0: "",  # Map from controller outputs to inputs used by ControlledCircle object
                1: "UP",  # First is for left player
                2: "UP RIGHT",
                3: "RIGHT",
                4: "DOWN RIGHT",
                5: "DOWN",
                6: "DOWN LEFT",
                7: "LEFT",
                8: "UP LEFT",
            },
            {
                0: "",  # Second map is for right player
                1: "UP",
                2: "UP LEFT",
                3: "LEFT",
                4: "DOWN LEFT",
                5: "DOWN",
                6: "DOWN RIGHT",
                7: "RIGHT",
                8: "UP RIGHT",
            },
        ]

        self.b = 20  # Buffer between walls and edge of world
        self.goal_width = 400
        self.goal_depth = 100
        self.puck = None  # Will contain the puck
        self.heuristic_flag = 0  # Used by heuristic controller for right agent
        self.score_scale = 10  # Scales the rewards
        self.num_cpu = 1  # Only one cpu is being trained here
        self.reset()

    def reset(self):
        """
        Reinitialize the world
        """
        self.obj_list = []
        self.heuristic_flag = 0  # Used by heuristic controller

        w, h = self.world_size
        b = self.b

        # The first set of objects define the outer wall of the world.
        gd = self.goal_depth
        # Top wall
        self.obj_list.append(
            StaticObject(LineShape, x0=b + gd, x1=w - b - gd, y0=h - b, y1=h - b, color=[255, 255, 255])
        )
        # Bottom wall
        self.obj_list.append(StaticObject(LineShape, x0=b + gd, x1=w - b - gd, y0=b, y1=b, color=[255, 255, 255]))

        self.obj_list.append(
            StaticObject(LineShape, x0=b + gd, x1=b + gd, y0=b, y1=h / 2 - self.goal_width / 2, color=[255, 255, 255])
        )  # Left wall Top
        self.obj_list.append(
            StaticObject(
                LineShape, x0=b + gd, x1=b + gd, y0=h - b, y1=h / 2 + self.goal_width / 2, color=[255, 255, 255]
            )
        )  # Left wall Bot
        self.obj_list.append(
            StaticObject(
                LineShape,
                x0=b + gd,
                x1=b,
                y0=h / 2 + self.goal_width / 2,
                y1=h / 2 + self.goal_width / 2,
                color=[80, 200, 255],
            )
        )  # Left goal top
        self.obj_list.append(
            StaticObject(
                LineShape,
                x0=b + gd,
                x1=b,
                y0=h / 2 - self.goal_width / 2,
                y1=h / 2 - self.goal_width / 2,
                color=[80, 200, 255],
            )
        )  # Left goal bot
        self.obj_list.append(
            StaticObject(
                LineShape,
                x0=b,
                x1=b,
                y0=h / 2 + self.goal_width / 2,
                y1=h / 2 - self.goal_width / 2,
                color=[80, 200, 255],
            )
        )  # Left goal back

        self.obj_list.append(
            StaticObject(
                LineShape, x0=w - b - gd, x1=w - b - gd, y0=b, y1=h / 2 - self.goal_width / 2, color=[255, 255, 255]
            )
        )  # Right wall Top
        self.obj_list.append(
            StaticObject(
                LineShape, x0=w - b - gd, x1=w - b - gd, y0=h - b, y1=h / 2 + self.goal_width / 2, color=[255, 255, 255]
            )
        )  # Right wall Bot
        self.obj_list.append(
            StaticObject(
                LineShape,
                x0=w - b - gd,
                x1=w - b,
                y0=h / 2 + self.goal_width / 2,
                y1=h / 2 + self.goal_width / 2,
                color=[255, 0, 0],
            )
        )  # Right goal top
        self.obj_list.append(
            StaticObject(
                LineShape,
                x0=w - b - gd,
                x1=w - b,
                y0=h / 2 - self.goal_width / 2,
                y1=h / 2 - self.goal_width / 2,
                color=[255, 0, 0],
            )
        )  # Right goal bot
        self.obj_list.append(
            StaticObject(
                LineShape,
                x0=w - b,
                x1=w - b,
                y0=h / 2 + self.goal_width / 2,
                y1=h / 2 - self.goal_width / 2,
                color=[255, 0, 0],
            )
        )  # Right goal back

        # Create players
        self.player_list = []

        # Player 1 (left agent)
        self.obj_list.append(
            ControlledCircle(
                CircleShape,
                x=[0.1 * w + gd, 0.5 * h],
                r=20,
                ang=0,
                mass=1,
                force_mag=1000,
                max_v=300,
                color=[80, 200, 255],
            )
        )
        self.player_list.append(
            Player(
                obj=self.obj_list[-1],
                score=0,
                last_action=0,
                control_func=self.cpu_controller.select_action,
                control_map=self.control_map[0],
            )
        )

        # Player 2 (right agent, uses a simple heuristic controller)
        self.obj_list.append(
            ControlledCircle(
                CircleShape,
                x=[0.9 * w - gd, 0.5 * h],
                r=20,
                ang=0,
                mass=1,
                force_mag=1000,
                max_v=300,
                color=[255, 0, 0],
            )
        )
        self.player_list.append(
            Player(
                obj=self.obj_list[-1],
                score=0,
                last_action=0,
                control_func=self.get_action_heuristic,
                control_map=self.control_map[1],
            )
        )

        # Create puck. y-position is random, as is an initial velocity.
        rand_v = np.random.rand(2) * 200 - 100
        self.puck = DynamicObject(
            shape=CircleShape,
            x=[0.5 * w, (0.25 + 0.5 * np.random.rand()) * h],
            r=40,
            ang=0,
            mass=0.2,
            max_v=400,
            color=[255, 255, 255],
            tau=17,
            v=rand_v,
        )
        self.obj_list.append(self.puck)

    def update_score(self):
        """
        Calculate a new score for this world. Scores for each agent are based on the distance of the puck to the
        opposing goal. A large score bonus or penalty is assigned when a goal is scored.
        """
        puck_x = self.puck.x
        w, h = self.world_size

        p1_dist_x = w - puck_x[0]
        p1_dist_y = np.maximum(0, np.abs(h / 2 - puck_x[1]) - (self.goal_width / 2 - self.puck.shape.r))
        p1_dist = np.sqrt(p1_dist_x ** 2 + p1_dist_y ** 2)

        p2_dist_x = puck_x[0]
        p2_dist_y = np.maximum(0, np.abs(h / 2 - puck_x[1]) - (self.goal_width / 2 - self.puck.shape.r))
        p2_dist = np.sqrt(p2_dist_x ** 2 + p2_dist_y ** 2)

        self.player_list[0].score = 0.5 - p1_dist / w
        self.player_list[1].score = 0.5 - p2_dist / w

        if puck_x[0] > w - self.b - self.goal_depth + self.puck.shape.r:  # Left player (p1) scored
            self.player_list[0].score = self.player_list[0].score + 1
            self.player_list[1].score = self.player_list[1].score - 1

        if puck_x[0] < self.b + self.goal_depth - self.puck.shape.r:  # Right player (p2) scored
            self.player_list[0].score = self.player_list[0].score - 1
            self.player_list[1].score = self.player_list[1].score + 1

        self.player_list[0].score = self.player_list[0].score * self.score_scale
        self.player_list[1].score = self.player_list[1].score * self.score_scale

    def terminate_run(self):
        """
        Run is terminated if either team scores a goal. The ball also occasionally escapes from the walls of the
        world due to an imperfect physics engine. The run is also terminated when this occurs.
        """
        puck_x = self.puck.x
        w, h = self.world_size

        # End if in goal, or if puck has escaped world.

        if (
            puck_x[0] > (w - self.b - self.goal_depth + self.puck.shape.r)
            or puck_x[0] < (self.b + self.goal_depth - self.puck.shape.r)
            or puck_x[1] < 0
            or puck_x[1] > h
        ):
            return True
        else:
            return False

    def get_state(self):

        """
        Get state for each player. This includes each player's position and velocity, the pucks position and velocity,
        and the player's position and velocity relative to the puck. Positions are shifted such that the (0,0)
        position is at the center of the world.

        Since the models are trained for the left team, the right team's states are modified such that it appears they
        are on the left team. Player 2's state won't be used by the heuristic controller.
        """
        ws = np.array(self.world_size)
        p1_state = np.zeros((1, 16))
        p1_state[0, 0:2] = self.player_list[0].obj.x - ws / 2
        p1_state[0, 2:4] = self.player_list[0].obj.v
        p1_state[0, 4:6] = self.player_list[1].obj.x - ws / 2
        p1_state[0, 6:8] = self.player_list[1].obj.v
        p1_state[0, 8:10] = self.puck.x - ws / 2
        p1_state[0, 10:12] = self.puck.v
        p1_state[0, 12:14] = self.player_list[0].obj.x - self.puck.x
        p1_state[0, 14:16] = self.player_list[0].obj.v - self.puck.v

        p2_state = np.zeros((1, 16))
        p2_state[0, 0:2] = self.player_list[1].obj.x - ws / 2
        p2_state[0, 2:4] = self.player_list[1].obj.v
        p2_state[0, 4:6] = self.player_list[0].obj.x - ws / 2
        p2_state[0, 6:8] = self.player_list[0].obj.v
        p2_state[0, 8:10] = self.puck.x - ws / 2
        p2_state[0, 10:12] = self.puck.v
        p2_state[0, 12:14] = self.player_list[1].obj.x - self.puck.x
        p2_state[0, 14:16] = self.player_list[1].obj.v - self.puck.v

        # "Swap sides" for player 2 so it appears as if its going left to right
        p2_state[0, ::2] = p2_state[0, ::2] * -1

        return [p1_state / 1000, p2_state / 1000]  # Scale values by size of world.

    def get_action_heuristic(self, state=None):
        """
        This is a heuristic control for player 2 (bot going right-to-left).

        State is unused. The agent will chase after the ball. If it founds itself ahead of the ball it will try to
        return to its own goal. Once behind the ball again it will chase.
        """
        p2_x = self.player_list[1].obj.x
        puck_x = self.puck.x
        w, h = self.world_size

        if p2_x[0] < puck_x[0] + 10:  # We've fallen ahead of the ball, set flag
            self.heuristic_flag = 1

        if p2_x[0] > puck_x[0] + 100:  # We're back behind, set flag
            self.heuristic_flag = 0

        if self.heuristic_flag == 1:  # In this case we're trying to move back into goal. May hit ball, hard to avoid
            if p2_x[1] > (h / 2 + self.goal_width / 2 - 20):
                inputs = 6  # DOWN RIGHT
            elif p2_x[1] < (h / 2 - self.goal_width / 2 + 20):
                inputs = 8  # UP RIGHT
            else:
                inputs = 7  # RIGHT
        else:  # We're behind ball, gonna move left toward it
            if p2_x[1] > puck_x[1] + 10:
                inputs = 4  # DOWN LEFT
            elif p2_x[1] < puck_x[1] - 10:
                inputs = 2  # UP LEFT
            else:
                inputs = 3  # LEFT

        return np.array([inputs])

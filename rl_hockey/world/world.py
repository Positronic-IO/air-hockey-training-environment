from typing import Any, Callable, Dict, List, Union

import numpy as np

from rl_hockey.controller import Controller
from rl_hockey.object.objects import StaticObject, DynamicObject, ControlledCircle
from rl_hockey.object.shapes import LineShape, CircleShape


class World:
    """
    Parent class for each world. Worlds are defined by:
    -A set of object instances (e.g., walls, balls, agents, etc)
    -One or more controllers that are trained to control one or more agents
    -If necessary, one or more controllers used to control opposing agents.

    Each world will also require several methods be defined in the child class:
    -reset() will reinitialize all objects in the world
    -update_score() will recalculate the score of each agent based on the state
    -terminate_run() is optional, will determine if the current world should be terminal (e.g., a goal is scored)
    -get_state() returns the state of each agent to be used by the controller
    -get_num_actions() returns the number of actions available to each agent
    """

    __slots__: List[str] = [
        "world_size",
        "obj_list",
        "player_list",
        "steps",
        "control_map",
        "cpu_controller",
        "opp_controller",
        "num_cpu",
    ]

    def __init__(self, world_size: List[int]):
        self.world_size = world_size
        self.obj_list: List[Union[StaticObject, DynamicObject, ControlledCircle, LineShape, CircleShape]] = list()
        self.player_list: List[ControlledCircle] = list()
        self.steps: int = 0
        self.cpu_controller: Controller = Controller()  # These will be set by set_cpu_controller.
        self.opp_controller: Controller = Controller()
        self.num_cpu: int = 1

        self.control_map: List[Dict[int, str]] = [
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

    def get_object_list(self):
        return self.obj_list

    def draw_world(self, high_scale=1):
        """
        This will render the world to an array, this array can then be drawn using Image.fromarray
        """
        hs_output_size = [int(w * high_scale) for w in self.world_size]
        hs_arr = np.zeros((*hs_output_size, 3))
        for o in self.obj_list:
            o.draw(hs_arr, high_scale)
        return hs_arr

    def get_world_size(self):
        """
        Get the size of the world
        """
        return self.world_size

    def get_num_actions(self):
        """
        Return number of actions available to each agent
        """
        return [len(cm) for cm in self.control_map]

    def set_cpu_controller(self, cpu_controller, opp_controller=None):
        """
        Set agent controllers, as well as opponent controllers if necessary
        """
        self.cpu_controller = cpu_controller
        self.opp_controller = opp_controller

    def get_num_cpu(self):
        """
        Return the number of controllers in training
        """
        return self.num_cpu

    def terminate_run(self):
        """
        Determine whether or not to terminate the current run. Will always return false unless redefined by child
        """
        return False

    def get_last_action(self):
        """
        Return the last action performed by each player agent
        """
        return [p.last_action for p in self.player_list]

    def get_scores(self):
        """
        Return the score of each player agent
        """
        return [p.score for p in self.player_list]

    def apply_control(self, state, frame_skip=False):
        """
        Each agent (player) is defined with a controller. This may be a model that is trained, a self play snapshot,
        or other heuristic control functions defined by the child class.

        This function calls each agent's controller using the input state, and then calls the appropriate object
        method to apply this action.

        When frame_skip=True the previously used action is used for each agent, rather than calling the controller.
        """
        # For each player in this World
        for player, s in zip(self.player_list, state):
            if frame_skip is False:  # If we are calling the controller this frame
                inputs = player.control_func(s)  # Get inputs from controller
                inputs = inputs.tolist()[0]
                player.last_action = inputs  # Record input for subsequent frames were controller isn't used
                inputs = player.control_map[inputs]  # Control map for this player
            else:  # Otherwise, use last action
                inputs = player.control_map[player.last_action]
            player.obj.apply_action(inputs)  # Apply this action to the object

        return

    # The following functions need to be defined by each World child class
    def reset(self):
        raise Exception("World child class needs to define reset() function")
        pass

    def update_score(self):
        raise Exception("World child class needs to define update_score() function")
        pass

    def get_state(self):
        raise Exception("World child class needs to define get_state() function")
        pass

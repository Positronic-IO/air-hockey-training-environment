""" Air Hockey Training Simulator """
import random

import numpy as np

from environment import AirHockey
from environment.test import TestAirHockey
from learners import LearnerFactory
from utils import get_model_path, parse_args, welcome


if __name__ == "__main__":
    # -------- Main Program Loop -----------

    # Parse args
    args = parse_args()

    # Welcome user
    welcome(args)

    # Initiate game environment
    if args.get("env") == "test":
        # Testing environment
        env = TestAirHockey()
    else:
        env = AirHockey()

    init = True
    learner = LearnerFactory().make(args["learner"], env)

    # If we pass a weights file, load it.
    if hasattr(learner, "load_model") and args.get("load"):
        file_name = get_model_path(args["load"])
        learner.load_model(file_name)

    # Epochs
    epoch = 1

    # Number of iterations between saves
    iterations_on_save = 10 ** 3
    iterations = 1

    # Game loop
    while True:
        
        # For first move, move in a random direction
        if init:
            actions = ["U", "D", "L", "R"]
            action = str(np.random.choice(actions))

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
                if args["learner"]  == "ddqn-learner":
                    next_state = (learner.location(), (env.puck.x, env.puck.y))
                    learner.remember(state, action, data["reward"], next_state)
                    learner.update()

                # DQN
                if args["learner"]  == "dqn-learner":
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


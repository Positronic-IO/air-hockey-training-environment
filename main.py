import json
import logging
import signal
import sys
import time
import tkinter as tk
from typing import Union

import numpy as np
import torch

from rl_hockey.controller import DDDQN, NaivePrioritizedBuffer, SelfPlayController
from rl_hockey.run import run
from rl_hockey.world import Hockey1v1, Hockey1v1Heuristic

# Initiate Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def signal_handler(sig, frame):
    """ Kill game gracefully """
    logger.info("Ending game")
    sys.exit(0)


torch.manual_seed(0)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--prev_model",
        type=str,
        default="./models/06_20_Hockey_1v1",
        help="If no previous model is set then a new one is created",
    )
    parser.add_argument("--num_iterations", type=int, default=200000, help="Number of iterations that will be run")
    parser.add_argument(
        "--train_steps", type=int, default=100, help="Number of training steps per controller, per iteration"
    )
    parser.add_argument("--draw_scale", type=float, default=0.5, help="Scale at which world will be drawn on screen")
    parser.add_argument("--eps_decay", type=int, default=500000, help="Decay rate of EPS in DQ")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final value of EP")
    parser.add_argument("--gamma", type=float, default=0.95, help="Gamma value used by DQ")
    parser.add_argument("--hn_size", type=int, default=512, help="Size of hidden node layers in mode")
    parser.add_argument("--mem_size", type=int, default=100000, help="Maximum size of memory")
    parser.add_argument("--learning_rate", type=float, default=2e-3, help="Learning rate of mode")
    parser.add_argument("--beta_start", type=float, default=0.4, help="Initial beta value used by prioritized memory")
    parser.add_argument("--beta_max", type=float, default=0.5, help="Final beta value")
    parser.add_argument("--beta_frames", type=int, default=1000, help="Transition period from start to max beta value")
    parser.add_argument("--view_run", action="store_true", help="Set to True to visualize the mode")
    parser.add_argument("--world", type=str, default="Hockey1v1Heuristic", help="World to use, available worlds")

    args = parser.parse_args()
    logger.info(f"Environment config:\n {json.dumps(vars(args), indent=2)}")  # Print config

    # Register ability to kill the game --- gracefully
    signal.signal(signal.SIGINT, signal_handler)

    def beta_by_frame(frame_idx: int) -> float:
        """ Beta used by the priorized memory """
        return min(args.beta_max, args.beta_start + frame_idx * (1.0 - args.beta_start) / args.beta_frames)

    world: Union[object, Hockey1v1, Hockey1v1Heuristic] = object

    # Set up world and required controllers.
    if args.world == "Hockey1v1":

        SAVE_NAME = "Hockey1v1"
        world = Hockey1v1()  # Create world
        SELF_PLAY = True  # This world using self play

    elif args.world == "Hockey1v1Heuristic":

        SAVE_NAME = "Hockey1v1Heuristic"
        world = Hockey1v1Heuristic()  # Create world
        SELF_PLAY = False  # World using an heuristic opponent

    else:
        raise Exception("World not recognized")

    memory = NaivePrioritizedBuffer(args.mem_size)  # Initialize empty memory
    num_actions = world.get_num_actions()  # Number of actions available for agent
    num_inputs = len(world.get_state()[0][0])  # Number of inputs provided
    cpu_controller = DDDQN(device="cuda")  # Create controller
    cpu_controller.create_model(
        num_inputs=num_inputs,
        num_actions=num_actions[0],
        gamma=args.gamma,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        lr=args.learning_rate,
        hn=args.hn_size,
    )

    # Currently if a args.prev_model is provided then it is not intended to be trained, but rather viewed.
    if args.prev_model:
        cpu_controller.load_model(args.prev_model + ".pt")  # Load model into controller
        cpu_controller.train_steps = args.eps_decay * 100  # Set train steps high so a low eps is used

        args.beta_start = args.beta_max  # Will result in beta always being equal to args.beta_max
        args.view_run = True  # Modify in future, currently can't properly load then train. Need to save memory

    if args.view_run:  # If a viewer is going to be used then setup the tk.Canvas
        world_size = world.get_world_size()
        print(world_size)
        root = tk.Tk()
        canvas = tk.Canvas(root, width=world_size[0] * args.draw_scale, height=world_size[1] * args.draw_scale)
        canvas.pack()

    # If self play is being used for training then we set up a SelfPlayController.
    # Occasional snapshots of the trained controller will be stored here to be used as an opponent
    if SELF_PLAY:  # If we're doing self play create controller and load initial model
        # Create a self play controller for each CPU controller
        self_play_cpu = SelfPlayController(num_actions=num_actions)
        self_play_cpu.insert_model(cpu_controller.get_model(), cpu_controller.get_eps())  # Load initial controller
    else:  # No self play, world will rely on it's own heuristic controllers
        self_play_cpu = None

    world.set_cpu_controller(cpu_controller, self_play_cpu)  # Insert controllers to world

    start = time.time()
    score_hist = np.zeros(args.num_iterations)  # Used to track score history

    for i in range(1, args.num_iterations):
        start = time.time()
        loss_mean = 0
        if args.view_run:  # If we are viewing the run pass the necessary arguments
            memory = run(memory, world=world, canvas=canvas, root=root, draw_step=2, pause_time=1 / 45, numSteps=20_000)
        else:
            memory = run(memory, world=world, numSteps=1500)  # Run an iteration
            for k in range(args.train_steps):  # Run a number of training steps
                loss_mean += cpu_controller.train(memory, beta_by_frame(i)) / args.train_steps

        stop = time.time()

        if SELF_PLAY is True:  # If self play is used, increment the internal counters
            self_play_cpu.increment_model()
            if i % 1000 == 0:  # Every 1000 iterations take a snapshot of controllers
                self_play_cpu.insert_model(cpu_controller.get_model(), cpu_controller.get_eps())

        if i % 100 == 0 and not args.view_run:  # Every 100 iterations save the current model(s)
            cpu_controller.save_model("./models/" + SAVE_NAME + ".pt")

        # Get current score and provide some output
        score_hist[i] = np.mean(world.get_scores()[: world.get_num_cpu()])
        score_mean = np.mean(score_hist[np.max([i - 100, 0]) : i + 1])
        logger.info(
            "Iteration {}, memLen {}, loss {:.6f}, time {:.2f}, score {}, avg_score {:.2f}".format(
                i, len(memory), loss_mean, stop - start, world.get_scores()[: world.get_num_cpu()], score_mean
            )
        )

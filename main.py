from rl_hockey.world import *
from rl_hockey.controller import DDDQN, NaivePrioritizedBuffer, SelfPlayController
import time
import tkinter as tk
import numpy as np
from rl_hockey.run import run

# BASE SETTINGS
PREVIOUS_MODEL = ""  # If no previous model is set then a new one is created
NUM_ITERATIONS = 200_000  # Number of iterations that will be run
TRAIN_STEPS_PER_TRIAL = 100  # Number of training steps per controller, per iteration
DRAW_SCALE = 0.5  # Scale at which world will be drawn on screen
EPS_DECAY = 500_000  # Decay rate of EPS in DQN
EPS_END = 0.05  # Final value of EPS
GAMMA = 0.95  # Gamma value used by DQN
HN_SIZE = 512  # Size of hidden node layers in model
MEM_SIZE = 100_000  # Maximum size of memory
LEARN_RATE = 2e-3  # Learning rate of model
BETA_START = 0.4  # Initial beta value used by prioritized memory
BETA_MAX = 0.5  # Final beta value
BETA_FRAMES = 1000  # Transition period from start to max veta value
VIEW_RUN = True  # Set to True to visualize the model
WORLD = "Hockey1v1Heuristic"  # World to use, available worlds:
# Hockey1v1, Hockey1v1Heuristic, Hockey2v2, Hockey2v2Roles

# Make changes to settings here
# Remove this line to train a new model, rather than visualizing a previous one
PREVIOUS_MODEL = "./models/06_20_Hockey_1v1"  # Do not include file extension on previous model

# beta used by the priorized memory
beta_by_frame = lambda frame_idx: min(BETA_MAX, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

if __name__ == "__main__":
    # Set up world and required controllers.
    if WORLD == "Hockey1v1":
        SAVE_NAME = "Hockey1v1"
        world = Hockey1v1()  # Create world
        SELF_PLAY = True  # This world using self play
        memory = [NaivePrioritizedBuffer(MEM_SIZE)]  # Initialize empty memory
        num_actions = world.get_num_actions()  # Number of actions available for agent
        num_inputs = len(world.get_state()[0][0])  # Number of inputs provided
        cpu_controller = [DDDQN(device="cuda")]  # Create controller
        cpu_controller[0].create_model(
            num_inputs=num_inputs,
            num_actions=num_actions[0],
            gamma=GAMMA,
            eps_end=EPS_END,
            eps_decay=EPS_DECAY,
            lr=LEARN_RATE,
            hn=HN_SIZE,
        )

    elif WORLD == "Hockey1v1Heuristic":
        SAVE_NAME = "Hockey1v1Heuristic"
        world = Hockey1v1Heuristic()  # Create world
        SELF_PLAY = False  # World using an heuristic opponent
        memory = [NaivePrioritizedBuffer(MEM_SIZE)]  # Initialize empty memory
        num_actions = world.get_num_actions()  # Number of actions available for agent
        num_inputs = len(world.get_state()[0][0])  # Number of inputs provided
        cpu_controller = [DDDQN(device="cuda")]  # Create controller
        cpu_controller[0].create_model(
            num_inputs=num_inputs,
            num_actions=num_actions[0],
            gamma=GAMMA,
            eps_end=EPS_END,
            eps_decay=EPS_DECAY,
            lr=LEARN_RATE,
            hn=HN_SIZE,
        )
    else:
        raise Exception("WORLD not recognized")

    # Currently if a PREVIOUS_MODEL is provided then it is not intended to be trained, but rather viewed.
    if PREVIOUS_MODEL is not "":
        if cpu_controller.count(cpu_controller[0]) == len(cpu_controller):  # If all CPU controllers are same
            cpu_controller[0].load_model(PREVIOUS_MODEL + ".pt")  # Load model into controller
            cpu_controller[0].train_steps = EPS_DECAY * 100  # Set train steps high so a low eps is used
        else:
            for i, c in enumerate(cpu_controller):  # If multiple CPU controllers are present
                c.load_model(PREVIOUS_MODEL + "_" + str(i) + ".pt")  # Load each model
                c.train_steps = EPS_DECAY * 100

        BETA_START = BETA_MAX  # Will result in beta always being equal to BETA_MAX
        VIEW_RUN = True  # Modify in future, currently can't properly load then train. Need to save memory

    if VIEW_RUN is True:  # If a viewer is going to be used then setup the tk.Canvas
        world_size = world.get_world_size()
        print(world_size)
        root = tk.Tk()
        canvas = tk.Canvas(root, width=world_size[0] * DRAW_SCALE, height=world_size[1] * DRAW_SCALE)
        canvas.pack()

    # If self play is being used for training then we set up a SelfPlayController.
    # Occasional snapshots of the trained controller will be stored here to be used as an opponent
    if SELF_PLAY:  # If we're doing self play create controller and load initial model
        self_play_cpu = []  # Create a self play controller for each CPU controller
        for c in cpu_controller:
            self_play_cpu.append(SelfPlayController(num_actions=num_actions))
            self_play_cpu[-1].insert_model(c.get_model(), c.get_eps())  # Load initial controller
    else:  # No self play, world will rely on it's own heuristic controllers
        self_play_cpu = None

    world.set_cpu_controller(cpu_controller, self_play_cpu)  # Insert controllers to world

    start = time.time()
    score_hist = np.zeros(NUM_ITERATIONS)  # Used to track score history

    for i in range(1, NUM_ITERATIONS):
        start = time.time()
        loss_mean = 0
        if VIEW_RUN:  # If we are viewing the run pass the necessary arguments
            memory = run(memory, world=world, canvas=canvas, root=root, draw_step=2, pause_time=1 / 45, numSteps=20_000)
        else:
            memory = run(memory, world=world, numSteps=1500)  # Run an iteration
            for j in range(len(cpu_controller)):  # For each CPU controller
                for k in range(TRAIN_STEPS_PER_TRIAL):  # Run a number of training steps
                    loss_mean += cpu_controller[j].train(memory[j], beta_by_frame(i)) / TRAIN_STEPS_PER_TRIAL

        stop = time.time()

        if SELF_PLAY is True:  # If self play is used, increment the internal counters
            for j in range(len(self_play_cpu)):  # of the self play controllers.
                self_play_cpu[j].increment_model()
                if i % 1000 == 0:  # Every 1000 iterations take a snapshot of controllers
                    self_play_cpu[j].insert_model(cpu_controller[j].get_model(), cpu_controller[j].get_eps())

        if i % 100 == 0 and VIEW_RUN is False:  # Every 100 iterations save the current model(s)
            if cpu_controller.count(cpu_controller[0]) == len(cpu_controller):
                cpu_controller[0].save_model("./models/" + SAVE_NAME + ".pt")
            else:
                for j, c in enumerate(cpu_controller):
                    c.save_model("./models/" + SAVE_NAME + "_" + str(j) + ".pt")

        # Get current score and provide some output
        score_hist[i] = np.mean(world.get_scores()[: world.get_num_cpu()])
        score_mean = np.mean(score_hist[np.max([i - 100, 0]) : i + 1])
        print(
            "Iteration {}, memLen {}, loss {:.6f}, time {:.2f}, score {}, avg_score {:.2f}".format(
                i, len(memory[0]), loss_mean, stop - start, world.get_scores()[: world.get_num_cpu()], score_mean
            )
        )

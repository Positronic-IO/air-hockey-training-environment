""" Dueling DDQN Config """

config = {
    "params": {
        "max_memory": 50000,
        "learning_rate": 0.00001,
        "gamma": 0.9,
        "epsilon": 1,
        "initial_epsilon": 1,
        "final_epsilon": 0.001,
        "batch_size": 10000,
        "observe": 5000,
        "explore": 50000,
        "frame_per_action": 4,
        "update_target_freq": 30000,
        "timestep_per_train": 10000,
        "iterations_on_save": 10000,
    },
    "load": "",
}

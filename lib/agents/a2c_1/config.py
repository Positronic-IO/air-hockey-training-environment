""" A2C 1 Config """

config = {
    "params": {
        "max_memory": 50000,
        "actor_learning_rate": 0.00001,
        "critic_learning_rate": 0.00001,
        "gamma": 0.95,
        "batch_size": 10000,
        "frame_per_action": 4,
        "timestep_per_train": 10000,
        "iterations_on_save": 10000,
        "epochs": 10,
        "epsilon": 1,
        "initial_epsilon": 1,
        "final_epsilon": 0.001,
        "observe": 5000,
        "explore": 50000,
    },
    "actor": {"load": ""},
    "critic": {"load": ""},
}

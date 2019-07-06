""" c51 Config """

config = {
    "params": {
        "max_memory": 50000,
        "learning_rate": 0.0001,
        "gamma": 0.95,
        "frame_per_action": 4,
        "batch_size": 10000,
        "update_target_freq": 30000,
        "timestep_per_train": 10000,
        "num_atoms": 10,
        "v_max": 10,
        "v_min": -10,
        "iterations_on_save": 10000,
        "epochs": 10,
    },
    "load": "",
}

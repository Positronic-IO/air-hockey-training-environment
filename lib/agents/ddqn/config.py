""" DDQN Config """

config = {
    "params": {
        "max_memory": 10000,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "batch_size": 10000,
        "sync_target_interval": 100000,
        "timestep_per_train": 100000,
    },
    "load": "",
}

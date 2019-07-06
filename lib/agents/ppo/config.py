""" PPO Config """

config = {
    "continuous": True,
    "params": {
        "max_memory": 5000,
        "actor_learning_rate": 0.00001,
        "critic_learning_rate": 0.00001,
        "gamma": 0.8,
        "batch_size": 5000,
        "epochs": 10,
        "timestep_per_train": 5000,
        "iterations_on_save": 5000,
        "noise": 1,
    },
    "actor": {"load": ""},
    "critic": {"load": ""},
}

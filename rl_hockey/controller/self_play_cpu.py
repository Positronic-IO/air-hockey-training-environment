import random

import torch
import copy

from .controller import Controller


class SelfPlayController(Controller):
    """
    When self play training is used periodic snapshots are taken of the model while it is training. The N most recent
    snapshots are stored in this class. The training model will then compete against one of these models each iteration.
    """

    def __init__(self, num_actions):
        super(SelfPlayController, self).__init__()
        self.device = torch.device("cuda")

        # These will get set when creating model
        self.num_actions = num_actions
        self.networks = []
        self.eps = []
        self.num_hist = 5  # Maximum number of models that will be stored
        self.next_remove = 0  # Next model to replace
        self.current_network = 0  # Next model to use

    def insert_model(self, model, eps=0.05):
        """
        Inserts the given model into the self play controller. If the maximum number of models are already stored here
        then the oldest model will be replaced.

        :param model: Model to be stored
        :param eps: current eps value of that model
        :return:
        """
        if len(self.networks) < self.num_hist:  # If we haven't maxed out copies then append model
            self.networks.append(copy.deepcopy(model))
            self.networks[-1].to(self.device)
            self.networks[-1].eval()
            self.eps.append(eps)
        else:
            self.networks[self.next_remove] = copy.deepcopy(model)
            self.networks[self.next_remove].to(self.device)
            self.networks[self.next_remove].eval()
            self.eps[self.next_remove] = eps
            self.next_remove = (self.next_remove + 1) % self.num_hist

        for n in self.networks:
            print(n.features[0].weight.sum())

    def increment_model(self):
        """
        Increment to use the next model in the next iteration
        """
        self.current_network = (self.current_network + 1) % len(self.networks)

    def select_action(self, state):
        """
        Select an action for the given state from the current self play model.
        """

        state = torch.tensor(state, dtype=torch.float32)
        sample = random.random()
        if sample > self.eps[self.current_network]:  # Use controller with chance (1-eps)
            with torch.no_grad():
                return self.networks[self.current_network](state.to(self.device)).max(1)[1]
        else:
            return torch.tensor(
                [random.randrange(self.networks[self.current_network].num_actions)],
                device=self.device,
                dtype=torch.long,
            )

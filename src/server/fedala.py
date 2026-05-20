from argparse import ArgumentParser, Namespace

from typing import Any, Dict
from omegaconf import DictConfig


from src.client.fedala import FedALAClient
from src.server.fedavg import FedAvgServer


class FedALAServer(FedAvgServer):
    algorithm_name = "FedALA"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedALAClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--layer_idx",
            type=int,
            default=2,
            help="Control the weight range. By default, all the layers are selected.",
        )
        parser.add_argument(
            "--num_pre_loss",
            type=int,
            default=10,
            help="The number of the recorded losses to be considered to calculate the standard deviation.",
        )
        parser.add_argument(
            "--eta", type=float, default=1.0, help="Weight learning rate."
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.1,
            help="Train the weight until the standard deviation of the recorded losses is less than a given threshold.",
        )
        parser.add_argument(
            "--rand_percent",
            type=float,
            default=0.8,
            help="The percent of the local training data to sample.",
        )
        return parser.parse_args(args_list)
    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.client_prev_model_states: Dict[int, Dict[str, Any]] = {}

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""

        client_packages = self.trainer.train()
        for client_id, package in client_packages.items():
            self.client_prev_model_states[client_id] = package["prev_model_state"]
        self.aggregate_client_updates(client_packages)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        if client_id in self.client_prev_model_states:
            server_package["prev_model_state"] = self.client_prev_model_states[
                client_id
            ]
        else:
            server_package["prev_model_state"] = None
        return server_package
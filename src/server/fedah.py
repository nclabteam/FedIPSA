from argparse import ArgumentParser, Namespace

from typing import Any, Dict

from omegaconf import DictConfig

from src.client.fedah import FedAHClient
from src.server.fedavg import FedAvgServer


class FedAHServer(FedAvgServer):
    algorithm_name = "FedAH"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedAHClient

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.client_prev_model_states: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument(
            "--eta", type=float, default=1.0, help="Weight learning rate."
        )
        parser.add_argument("--plocal_epochs", type=int, default=1)
        return parser.parse_args(args_list)

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
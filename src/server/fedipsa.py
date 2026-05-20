from argparse import ArgumentParser, Namespace
from typing import Any, Dict
from collections import OrderedDict

from omegaconf import DictConfig

from src.client.fedipsa import FedIPSAClient
from src.server.fedavg import FedAvgServer
import torch
import copy

class FedIPSAServer(FedAvgServer):
    algorithm_name = "FedIPSA"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedIPSAClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--growth", type=float, default=0.1)
        parser.add_argument("--bound", type=float, default=0.75)
        parser.add_argument("--interval", type=int, default=5)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.clients_round = [0 for _ in range(self.client_num)]
        self.client_prev_model_states: Dict[int, Dict[str, Any]] = {}
        self.client_masks: Dict[int, Dict[str, Any]] = {}

    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Aggregate clients model parameters and produce global model
        parameters.

        Args:
            client_packages: Dict of client parameter packages, with format:
            {
                `client_id`: {
                    `regular_model_params`: ...,
                    `optimizer_state`: ...,
                }
            }

            About the content of client parameter package, check `FedAvgClient.package()`.
        """

        # super().aggregate_client_updates(client_packages)
        client_weights = [package["weight"] for package in client_packages.values()]

        for name, global_param in self.public_model_params.items():
            client_params = []
            drop_id = []
            for id, package in enumerate(client_packages.values()):
                if name in package["regular_model_params"].keys():
                    if not torch.isnan(package["regular_model_params"][name]).any():
                        client_params.append(package["regular_model_params"][name])
                    else:
                        drop_id.append(id)
                        continue

            if client_params == []:
                continue
            weights = copy.copy(client_weights)
            if len(drop_id) > 0:
                for id in reversed(drop_id):
                    weights.pop(id)

            weights = torch.tensor(weights) / sum(weights)
            
            client_params = torch.stack(client_params, dim=-1)
            aggregated = torch.sum(client_params * weights, dim=-1)

            global_param.data = aggregated
        self.model.load_state_dict(self.public_model_params, strict=False)


        for cid, package in client_packages.items():
            self.clients_round[cid] = package['client_round']
            self.client_prev_model_states[cid] = package["prev_model_state"]
            self.client_masks[cid] = package['mask']

    def package(self, client_id: int):
        """Package parameters that the client-side training needs. If you are
        implementing your own FL method and your method has different
        parameters to FedAvg's that passes from server-side to client-side,
        this method need to be overrided. All this method should do is
        returning a dict that contains all parameters.

        Args:
            client_id: The client ID.

        Returns:
            A dict of parameters: {
                `client_id`: The client ID.
                `local_epoch`: The num of epoches that client local training performs.
                `client_model_params`: The client model parameter dict.
                `optimizer_state`: The client model optimizer's state dict.
                `lr_scheduler_state`: The client learning scheduler's state dict.
                `return_diff`: Flag that indicates whether client should send parameters difference.
                    `False`: Client sends vanilla model parameters;
                    `True`: Client sends `diff = global - local`.
            }.
        """
        server_package = super().package(client_id)
        server_package['client_round'] = self.clients_round[client_id]
        if client_id in self.client_masks.keys():
            server_package['mask'] = self.client_masks[client_id]
        else:
            server_package['mask'] = None
        if client_id in self.client_prev_model_states:
            server_package["prev_model_state"] = self.client_prev_model_states[
                client_id
            ]
        else:
            server_package["prev_model_state"] = None

        return server_package   
    

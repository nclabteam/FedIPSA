from argparse import ArgumentParser, Namespace
from typing import Any, Dict
from collections import OrderedDict

from omegaconf import DictConfig
from src.client.fedhkdsa import FedHKDSAClient
from src.server.fedavg import FedAvgServer
from copy import deepcopy

import torch

class FedHKDSAServer(FedAvgServer):
    algorithm_name = "FedHKDSA"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedHKDSAClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--lamda", type=float, default=1)
        parser.add_argument("--tau", type=float, default=1)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.client_params = {}
        self.client_prev_model_states: Dict[int, Dict[str, Any]] = {}
    
    def aggregate_client_updates(
        self, client_packages
    ):
        # first aggregation
        client_weights = [package["weight"] for package in client_packages.values()]
        weights = torch.tensor(client_weights) / sum(client_weights)
        if self.return_diff:  # inputs are model params diff
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        package["model_params_diff"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1)
                self.public_model_params[name].data -= aggregated
        else:
            for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(client_params * weights, dim=-1)

                global_param.data = aggregated

        # cosine similarity
        sim = torch.zeros(len(client_packages))
        for idx, (_, package) in enumerate(client_packages.items()):
            for name in self.public_model_params.keys():
                sim[idx] += torch.sum((self.public_model_params[name] * package["regular_model_params"][name] / 
                (torch.norm(self.public_model_params[name]) * torch.norm(package["regular_model_params"][name]) + 1e-8)))

        norm_sim = sim / torch.sum(sim)
        
        # secondary aggregation
        for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        package["regular_model_params"][name]
                        for package in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(client_params * norm_sim, dim=-1)

                global_param.data = aggregated
        self.model.load_state_dict(self.public_model_params, strict=False)

        for cid, package in client_packages.items():
            self.client_params[cid] = package['regular_model_params'] 
            self.client_prev_model_states[cid] = package["prev_model_state"]


    def get_client_model_params(self, client_id: int):
        """This function is for outputting model parameters that asked by
        `client_id`.

        Args:
            client_id (int): The ID of query client.

        Returns:
            {
                `regular_model_params`: Generally model parameters that join aggregation.
                `personal_model_params`: Client personal model parameters that won't join aggregation.
            }
        """
        if client_id in self.client_params:
            regular_params = self.client_params[client_id]
        else:
            regular_params = deepcopy(self.public_model_params)
        personal_params = self.clients_personal_model_params[client_id]
        return dict(
            regular_model_params=regular_params, personal_model_params=personal_params
        )

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
        if client_id in self.client_prev_model_states:
            server_package["prev_model_state"] = self.client_prev_model_states[
                client_id
            ]
        else:
            server_package["prev_model_state"] = None

        return server_package
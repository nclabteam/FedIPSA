from typing import Any

import torch.nn.functional as F
import numpy as np

import copy
import torch

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES
from src.utils.models import DecoupledModel
from collections import defaultdict


class FedIPSAClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.growth = self.args.fedipsa.growth
        self.bound = self.args.fedipsa.bound
        self.interval = self.args.fedipsa.interval
        self.lr = self.args.optimizer.lr
        self.client_round = 0
        self.params_mask = {}
        self.si = {}
        self.num_params = sum(p.numel() for p in self.model.parameters())
        # self.personal_params_name.extend(
        #     [name for name in self.model.state_dict().keys() if "classifier" in name]
        # )
        self.prev_model: DecoupledModel = copy.deepcopy(self.model)


    def fit(self):
        self.model.train()
        self.dataset.train()
        if (self.interval > 0 and self.client_round > 1 and self.client_round % self.interval == 0) and self.get_cutoff_idx() != None:
            si_calc = True
            omega = defaultdict(float)
            # theta_old = {name: p.clone().detach() for name, p in self.model.named_parameters()}
            delta_theta = defaultdict(float)
        else:
            si_calc = False
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if si_calc:
                    with torch.no_grad():
                        for name, p in self.model.named_parameters():
                            if name not in self.personal_params_name:
                                # delta = p.detach() - theta_old[name]
                                delta = p.grad * self.lr
                                omega[name] += -p.grad * delta
                                delta_theta[name] += delta.pow(2)
                        
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(metrics=loss)
                else:
                    self.lr_scheduler.step()
        if si_calc:
            self.si = {
                name: omega[name] / (delta_theta[name] + 1e-8)
                for name in omega
            }
            self.params_selection()
        self.client_round += 1

    
    def params_selection(self):
  
        cutoff_idx = self.get_cutoff_idx()
        if cutoff_idx != None:
            si_values = torch.Tensor([]).to(self.device)
            for f in self.si.values():
                si_values = torch.cat([si_values, torch.flatten(f.clone().detach())], dim=0)
            self.params_mask = {}
            sorted_si, _ = torch.sort(si_values, descending=True)
            cutoff = sorted_si[cutoff_idx].item()
            for name, value in self.si.items():
                if any(torch.flatten(value >= cutoff)):
                    self.params_mask[name] = (value < cutoff).int()
            # print(f'client {self.client_id} num_join_round {self.client_round}, personalized params: {cutoff_idx}/{self.num_params}.')
        

    def get_cutoff_idx(self):
        
        cutoff_idx = int((self.client_round / self.interval) * self.growth * self.num_params)
        if cutoff_idx > int(self.bound * self.num_params):
            if cutoff_idx - int(self.bound * self.num_params) < int(self.growth * self.num_params):
                cutoff_idx = int(self.bound * self.num_params)
            else:
                return None
        return cutoff_idx

    
    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        self.client_round = package['client_round']
        if self.client_round != 0:
            self.personal_params_name = [name for name in package['personal_model_params'].keys()] 
        self.regular_params_name = []
        for name in package['regular_model_params'].keys():
            if name not in self.personal_params_name:
                self.regular_params_name.append(name)
        
        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])
            if package['mask']:
                self.params_mask = package['mask']

        else:
            self.prev_model.load_state_dict(self.model.state_dict())
        if not self.testing:
            self.align_params()
        else:
            # evaluates clients' personalized models
            self.model.load_state_dict(self.prev_model.state_dict())
    
    def package(self):
        client_package = super().package()
        client_package["mask"] = self.params_mask
        client_package['client_round'] = self.client_round
        client_package["prev_model_state"] = copy.deepcopy(self.model.state_dict())
        return client_package
    
    def align_params(self):
        self.prev_model.eval()
        self.prev_model.to(self.device)
        self.model.train()
        self.dataset.train()

        align_params = [
            param for name, param in self.model.named_parameters()
            if not name in self.personal_params_name
        ]
        alignment_optimizer = torch.optim.SGD(
            align_params, lr=0.05
        )
        class_features = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]

        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.prev_model.get_last_features(x)

                for y, feat in zip(y, features):
                    class_features[y].append(feat)

        mean_class_features = [
            torch.stack(f).mean(dim=0) if f else None
            for f in class_features
        ]

        for _ in range(1):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.get_last_features(x, detach=False)
                loss = 0
                for label in y.unique().tolist():
                    if mean_class_features[label] is not None:
                        loss += torch.mean(torch.abs(
                            features[y == label].mean(dim=0) - mean_class_features[label]
                        ))
                alignment_optimizer.zero_grad()
                loss.backward()
                alignment_optimizer.step()

        if self.params_mask:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.params_mask.keys():
                        param = param * self.params_mask[name].to(self.device) + self.prev_model.state_dict()[name].to(self.device) * (1 - self.params_mask[name].to(self.device))
        self.prev_model.cpu()

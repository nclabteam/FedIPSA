from src.client.fedavg import FedAvgClient
import torch
import torch.nn.functional as F

from src.utils.constants import NUM_CLASSES
from src.utils.models import DecoupledModel
from copy import deepcopy


class FedHKDSAClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.prev_model: DecoupledModel = deepcopy(self.model)
        self.first_round = False

    def set_parameters(self, package):
        super().set_parameters(package)
        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])
            self.first_round = False
        else:
            self.prev_model.load_state_dict(self.model.state_dict())
            self.first_round = True


    def fit(self):
        self.model.train()
        self.dataset.train()

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                if not self.first_round:
                    kd_loss = (self.args.fedhkdsa.tau**2 * torch.sum(torch.softmax(self.prev_model(x), dim=-1)) * 
                            torch.log(torch.sum(torch.softmax(self.prev_model(x), dim=-1)) / torch.sum(torch.softmax(logit, dim=-1)) + 1e-8))
                    loss = self.criterion(logit, y) + self.args.fedhkdsa.lamda * kd_loss
                    # print(loss, kd_loss)
                else:
                    loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(metrics=loss)
                else:
                    self.lr_scheduler.step()

    def package(self):
        client_package = super().package()
        client_package["prev_model_state"] = deepcopy(self.model.state_dict())
        return client_package
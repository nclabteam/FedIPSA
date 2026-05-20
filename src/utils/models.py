from collections import OrderedDict
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from .resnet import resnet4, resnet6, resnet8, resnet10, resnet18, resnet34, resnet50, resnet101, resnet152, resnet1d4, resnet1d10, resnet1d18
from omegaconf import DictConfig
from torch import Tensor

from src.utils.constants import DATA_SHAPE, INPUT_CHANNELS, NUM_CLASSES


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: list[nn.Module] = []
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        self.device = device
        return super().to(device)

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def _get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.detach().clone())

        for module in target_modules:
            module.register_forward_hook(_get_feature_hook_fn)

    def check_and_preprocess(self, args: DictConfig):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module for module in self.modules() if isinstance(module, nn.Dropout)
        ]
        if args.common.buffers == "global":
            for module in self.modules():
                if isinstance(
                    module,
                    (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d),
                ):
                    buffers_list = list(module.named_buffers())
                    for name_buffer, buffer in buffers_list:
                        # transform buffer to parameter
                        # for showing out in model.parameters()
                        delattr(module, name_buffer)
                        module.register_parameter(
                            name_buffer,
                            torch.nn.Parameter(buffer.float(), requires_grad=False),
                        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))

    def get_last_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()
        func = (lambda x: x.detach().clone()) if detach else (lambda x: x)
        try:
            out = self.base(x)
        except RuntimeError as err:
            if x.shape[1] == 1:
                x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
                try:
                    out = self.base(x)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Seems {self.__class__.__name__} does not support this dataset. Data resizing may help."
                    ) from err
            else:
                raise RuntimeError(
                    f"Seems {self.__class__.__name__} does not support this dataset."
                ) from err
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[list[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        try:
            _ = self.base(x)
        except RuntimeError as err:
            if x.shape[1] == 1:
                x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
                try:
                    _ = self.base(x)
                except RuntimeError as err:
                    raise RuntimeError(
                        f"Seems {self.__class__.__name__} does not support this dataset. Data resizing may help."
                    ) from err
            else:
                raise RuntimeError(
                    f"Seems {self.__class__.__name__} does not support this dataset."
                ) from err
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


# CNN used in FedAvg
class FedAvgCNN(DecoupledModel):
    feature_length = {
        "femnist": 1024,
        "cifar10": 1600,
        "cifar100": 1600,
    }

    def __init__(self, dataset: str, pretrained):
        super(FedAvgCNN, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 32, 5),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(32, 64, 5),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 512),
                activation3=nn.ReLU(),
            )
        )
        self.classifier = nn.Linear(512, NUM_CLASSES[dataset])


class LeNet5(DecoupledModel):
    feature_length = {
        "femnist": 256,
        "cifar10": 400,
        "cifar100": 400,
    }

    def __init__(self, dataset: str, pretrained):
        super(LeNet5, self).__init__()
        self.base = nn.Sequential(
            OrderedDict(
                conv1=nn.Conv2d(INPUT_CHANNELS[dataset], 6, 5),
                bn1=nn.BatchNorm2d(6),
                activation1=nn.ReLU(),
                pool1=nn.MaxPool2d(2),
                conv2=nn.Conv2d(6, 16, 5),
                bn2=nn.BatchNorm2d(16),
                activation2=nn.ReLU(),
                pool2=nn.MaxPool2d(2),
                flatten=nn.Flatten(),
                fc1=nn.Linear(self.feature_length[dataset], 120),
                activation3=nn.ReLU(),
                fc2=nn.Linear(120, 84),
                activation4=nn.ReLU(),
            )
        )

        self.classifier = nn.Linear(84, NUM_CLASSES[dataset])

class ResNet(DecoupledModel):
    archs = {
        "4": resnet4,
        "10": resnet10,
        "18": resnet18,
        "34": resnet34,
        "50": resnet50,
        "101": resnet101,
        "152": resnet152,
    }

    def __init__(self, version, dataset, pretrained):
        super().__init__()

        resnet = self.archs[version]()
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # if input is grayscale, repeat it to 3 channels
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        return super().forward(x)

class LSTMNet(DecoupledModel):
    def __init__(self, dataset, pretrained):
        super().__init__()
        self.base = LSTMNetBase(hidden_dim=128, num_layers=2, bidirectional=False, dropout=0.2,
                                padding_idx=0, vocab_size=80)
        self.classifier = nn.Linear(128, NUM_CLASSES[dataset])

    def get_last_features(self, x: Tensor, detach=True) -> Tensor:
        self.base.dropout.eval()
        out = self.base(x)
        return out
        # super().get_last_features(x, detach)

class LSTMNetBase(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=False, dropout=0.2, 
                padding_idx=0, vocab_size=80):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        self.lstm = nn.LSTM(input_size=hidden_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        # dims = hidden_dim*2 if bidirectional else hidden_dim
        # self.fc = nn.Linear(dims, num_classes)

    def forward(self, x):
        if type(x) == type([]):
            text, text_lengths = x
        else:
            text, text_lengths = x, [x.shape[1] for _ in range(x.shape[0])]
        
        embedded = self.embedding(text)
        
        #pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        #unpack sequence
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        out = torch.relu_(out[:,-1,:])
        out = self.dropout(out)
        # out = self.fc(out)
        # out = F.log_softmax(out, dim=1)
            
        return out

class HARLinear(DecoupledModel):
    def __init__(self, dataset, pretrained):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(561, 256), 
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 32),
                                   nn.ReLU()
                                   )
        self.classifier = nn.Linear(32, NUM_CLASSES[dataset])

class ResNet1d(DecoupledModel):
    archs = {
        "4": resnet1d4,
        "10": resnet1d10,
        "18": resnet1d18,
    }

    def __init__(self, version, dataset, pretrained):
        super().__init__()

        resnet = self.archs[version]()
        self.base = resnet
        self.classifier = nn.Linear(self.base.fc.in_features, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(-1, -2)
        return super().forward(x)
    def get_last_features(self, x, detach=True):
        x = x.transpose(-1, -2)
        return super().get_last_features(x, detach)
# NOTE: You can build your custom model here.
# What you only need to do is define the architecture in __init__().
# Don't need to consider anything else, which are handled by DecoupledModel well already.
# Run `python *.py -m custom` to use your custom model.
class CustomModel(DecoupledModel):
    def __init__(self, dataset):
        super().__init__()
        # You need to define:
        # 1. self.base (the feature extractor part)
        # 2. self.classifier (normally the final fully connected layer)
        # The default forwarding process is: out = self.classifier(self.base(input))
        pass


MODELS = {
    "custom": CustomModel,
    "lenet5": LeNet5,
    "avgcnn": FedAvgCNN,
    "res4": partial(ResNet, version="4"),
    "res10": partial(ResNet, version="10"),
    "res18": partial(ResNet, version="18"),
    "res34": partial(ResNet, version="34"),
    "lstmnet": LSTMNet,
    "harlinear": HARLinear,
    "res1d4": partial(ResNet1d, version="4"),
    "res1d10": partial(ResNet1d, version="10"),
    "res1d18": partial(ResNet1d, version="18"),
}

import torch
from environments.base_environment import *
from torchvision import datasets, transforms

from models.ConvArch import ConvArch
from models.MLP import MLP
from models.resnet import resnet18
import wandb
import numpy as np

class cifar10_Conv_env(BaseEnvironment):
    def __init__(self):
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.247, 0.2434, 0.2615)

    def get_dataset(self):
        dataset = datasets.CIFAR10(root=r'cifar10',
                                   train=True,
                                   transform=self.get_train_transform(),
                                   download=True)
        return dataset

    def get_test_dataset(self):
        dataset = datasets.CIFAR10(root=r'cifar10',
                                   train=False,
                                   transform=self.get_eval_transform(),
                                   download=True)
        return dataset

    def get_eval_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])
        return transform

    def get_train_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])
        return transform

    def get_layers(self, model):
        layers = model.secondary_layers
        return layers

    def get_model(self, **kwargs):
        depth = 6
        width = 50

        input_image_width = self.train_dataset[0][0].shape[1]
        model = ConvArch(input_channel_number=3, input_image_width=input_image_width, num_classes=10, depth=depth,
                         width=width)

        if self.use_cuda:
            model = model.cuda()
        wandb.config.update({"model": "CIFAR10 Conv", "depth": depth, "width": width})
        return model

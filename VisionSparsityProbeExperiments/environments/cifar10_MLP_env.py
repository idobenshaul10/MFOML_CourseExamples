import torch
from environments.base_environment import *
from torchvision import datasets, transforms

from models.MLP import MLP
from models.resnet import resnet18
import wandb
import numpy as np

class cifar10_MLP_env(BaseEnvironment):
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
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])
        return transform

    def get_layers(self, model):
        return model.layers

    def get_model(self, **kwargs):
        depth = 10
        width = 500
        input_shape = int(np.prod(self.train_dataset[0][0].shape))
        model = MLP(input_size=input_shape, num_classes=10, depth=depth, width=width)
        if self.use_cuda:
            model = model.cuda()
        wandb.config.update({"model": "CIFAR10 MLP", "depth": depth, "width": width})
        return model

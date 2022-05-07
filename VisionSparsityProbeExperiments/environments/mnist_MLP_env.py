import torch
from environments.base_environment import *
from torchvision import datasets, transforms

from models.MLP import MLP
from models.resnet import resnet18
import wandb


class mnist_MLP_env(BaseEnvironment):
    def __init__(self):
        super().__init__()

    def get_dataset(self):
        dataset = datasets.MNIST(root=r'/home/ido/datasets/MNIST',
                                 train=True,
                                 transform=self.get_eval_transform(),
                                 download=True)
        return dataset

    def get_test_dataset(self):
        dataset = datasets.MNIST(root=r'/home/ido/datasets/MNIST',
                                 train=False,
                                 transform=self.get_eval_transform(),
                                 download=True)
        return dataset

    def get_eval_transform(self):
        mean = (0.1307)
        std = (0.3081)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        return transform

    def get_layers(self, model):
        return model.layers

    def get_model(self, **kwargs):
        depth = 4
        width = 100
        model = MLP(input_size=28 * 28, num_classes=10, depth=depth, width=width)
        if self.use_cuda:
            model = model.cuda()
        wandb.config.update({"model": "MNIST MLP", "depth": depth, "width": width})
        return model

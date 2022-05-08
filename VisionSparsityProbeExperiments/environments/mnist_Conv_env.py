from environments.base_environment import *
from torchvision import datasets, transforms
from models.ConvArch import ConvArch
import wandb
import torch

class mnist_Conv_env(BaseEnvironment):
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
        layers = model.secondary_layers
        return layers

    def get_model(self, **kwargs):
        depth = 5
        width = 5

        input_image_width = self.train_dataset[0][0].shape[1]
        model = ConvArch(input_channel_number=1, input_image_width=input_image_width, num_classes=10, depth=depth,
                         width=width)

        if self.use_cuda:
            model = model.cuda()
        # checkpoint = torch.load(r'checkpoints/weights.best.h5')['checkpoint']
        # model.load_state_dict(checkpoint)
        wandb.config.update({"model": "MNIST Conv", "depth": depth, "width": width})
        return model

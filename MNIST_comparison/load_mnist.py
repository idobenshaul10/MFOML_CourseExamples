import torch
from dataclasses import dataclass
from torchvision import datasets, transforms
import gc

from ConvNet import CNN


@dataclass
class MNISTEnvironment:
    use_cuda: bool = False

    def __post_init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.mean = (0.1307)
        self.std = (0.3081)

    def load_environment(self, **kwargs):
        train_dataset = self.get_dataset()
        test_dataset = None
        try:
            test_dataset = self.get_test_dataset()
        except:
            print("test_dataset not available!")
        model = None
        model_cpu = self.get_model()
        if torch.cuda.is_available():
            model = model_cpu.cuda()

        layers = self.get_layers(model)

        return model, train_dataset, test_dataset, layers

    def get_test_dataset(self):
        dataset = datasets.MNIST(root=r'/home/ido/datasets/mnist/MNIST/MNIST/raw',
                                 train=False,
                                 transform=self.get_eval_transform(),
                                 download=True)
        return dataset

    def get_layers(self, model):
        pass

    def get_dataset(self):
        dataset = datasets.MNIST(root=r'mnist',
                                   train=True,
                                   transform=self.get_eval_transform(),
                                   download=True)
        return dataset

    def get_eval_transform(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])
        return transform

    def get_model(self):
        return CNN()

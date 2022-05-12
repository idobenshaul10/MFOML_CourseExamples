from environments.base_environment import *
from torchvision import datasets, transforms
from models.ConvArch import ConvArch
import wandb
import torch
import requests
import pickle
from torch.utils.data import TensorDataset, DataLoader

from models.ConvArch_1D import ConvArch_1D

class mnist_1D_Conv_env(BaseEnvironment):
    def __init__(self):
        super().__init__()
        url = 'https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl'
        r = requests.get(url, allow_redirects=True)
        open('./mnist1d_data.pkl', 'wb').write(r.content)
        with open('./mnist1d_data.pkl', 'rb') as handle:
            self.data = pickle.load(handle)

    def get_dataset(self):
        x_train = torch.Tensor(self.data['x'])
        y_train = torch.LongTensor(self.data['y'])
        dataset = TensorDataset(x_train, y_train)
        return dataset

    def get_test_dataset(self):
        x_test = torch.Tensor(self.data['x_test'])
        y_test = torch.LongTensor(self.data['y_test'])
        dataset = TensorDataset(x_test, y_test)
        return dataset

    def get_layers(self, model):
        layers = model.initial_layers + model.secondary_layers[:-2]
        return layers

    def get_model(self, **kwargs):
        depth = 5
        width = 12
        # input_image_width = self.train_dataset[0][0].shape[1]
        input_size   = self.train_dataset[0][0].shape[0]
        model = ConvArch_1D(input_size=input_size, num_classes=10, depth=depth,
                         width=width)

        if self.use_cuda:
            model = model.cuda()
        # checkpoint = torch.load(r'checkpoints/weights.best.h5')['checkpoint']
        # model.load_state_dict(checkpoint)
        wandb.config.update({"model": "MNIST Conv", "depth": depth, "width": width})
        return model

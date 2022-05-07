import torch
import torch.nn as nn
from dataclasses import dataclass
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class MLPLayer(nn.Module):
	def __init__(self, input_size: int, output_size: int):
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.linear = nn.Linear(self.input_size, self.output_size)
		self.bn = nn.BatchNorm1d(self.output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.linear(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class MLP(nn.Module):
	def __init__(self, input_size: int, num_classes: int, depth: int, width: int):
		super().__init__()
		self.input_size = input_size
		self.num_classes = num_classes
		self.depth = depth
		self.width = width
		self.init_layers()

	def init_layers(self):
		layers = [MLPLayer(self.input_size, self.width)]
		for _ in range(self.depth - 2):
			layers.append(MLPLayer(self.width, self.width))
		layers.append(nn.Linear(self.width, self.num_classes))
		self.layers = nn.ModuleList(layers)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		for layer in self.layers:
			x = layer(x)
		return x


def get_eval_transform():
	mean = (0.1307)
	std = (0.3081)
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)])
	return transform


if __name__ == '__main__':
	dataset = datasets.MNIST(root=r'/home/ido/datasets/MNIST',
							train=True,
							transform=get_eval_transform(),
							download=True)

	input_size = int(np.prod(dataset[0][0].shape))
	model = MLP(input_size=input_size, num_classes=10, depth=5, width=100)
	train_loader = DataLoader(dataset=dataset,
							  batch_size=128,
							  shuffle=True)

	for idx, data in enumerate(train_loader):
		x, y = data
		output = model(x)
		print(output.shape)
		break
	# model(dataset[i][0] for i in ragne)
	# import pdb; pdb.set_trace()
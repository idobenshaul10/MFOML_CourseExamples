import torch
import torch.nn as nn
from dataclasses import dataclass
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class ConvLayer_1D(nn.Module):
	def __init__(self, input_size: int, output_size: int):
		super().__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.conv = nn.Conv1d(self.input_size, self.output_size, kernel_size=3, stride=1, padding=1)
		self.bn = nn.BatchNorm1d(self.output_size)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv(x)
		x = self.relu(x)
		return x


class ConvArch_1D(nn.Module):
	def __init__(self, input_size: int, num_classes: int, depth: int, width: int):
		super().__init__()
		self.input_size = input_size
		self.num_classes = num_classes
		self.depth = depth
		self.width = width
		self.init_layers()

	def get_initial_layers(self):
		layers = [nn.Conv1d(1, self.width, kernel_size=2, stride=2, padding=0)]
		layers.append(nn.Conv1d(self.width, self.width, kernel_size=2, stride=2, padding=0))
		layers.append(nn.BatchNorm1d(self.width))
		layers.append(nn.ReLU())
		return layers

	def init_layers(self):
		self.initial_layers = nn.ModuleList(self.get_initial_layers())
		layers = []
		for _ in range(self.depth):
			layers.append(ConvLayer_1D(self.width, self.width))
		self.secondary_layers = nn.ModuleList(layers)
		self.fc = nn.Linear(self.width*self.input_size//4, self.num_classes)

	def forward(self, orig):
		x = orig.unsqueeze(1)
		for layer in self.initial_layers:
			x = layer(x)
		first_layer_output = x
		for layer in self.secondary_layers:
			x = layer(x)
		second_layer_output = x
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

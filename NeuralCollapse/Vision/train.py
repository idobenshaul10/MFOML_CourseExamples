from tqdm import tqdm
from dataclasses import dataclass
import torch
import torch.nn as nn
import gc
import torch.functional as F
import torch.optim as optim
from losses import ConsistancyLoss
from torchvision import datasets, transforms

class TrainFeatures:
	pass

@dataclass
class Trainer:
	conf: dict
	model: nn.Module
	optimizer: optim
	criterion: nn.Module
	train_loader: torch.utils.data.DataLoader
	num_classes: int
	device: torch.device
	epochs: int
	debug: bool = True
	use_consistency_loss: bool = False
	layers: list = None
	consis_criterion: nn.Module = None
	distributed: bool = False
	next_parameters: dict = None


	def __post_init__(self):
		self.batch_size = self.conf['batch_size']
		if self.use_consistency_loss:
			print(f"Inited trainer, using consistency loss")
			self.train_features = TrainFeatures()
			alpha_coef = self.next_parameters['alpha_consis']
			self.consis_criterion = ConsistancyLoss(num_layers=len(self.layers), num_classes=self.num_classes, alpha_coef=alpha_coef)


	def get_hook(self):
		def hook(model, input, output):
			self.train_features.value = input[0].clone()
		return hook


	def train(self, epoch:int):
		self.model.train()

		pbar = tqdm(total=len(self.train_loader), position=0, leave=True)
		for batch_idx, (data, target) in enumerate(self.train_loader, start=1):
			if data.shape[0] != self.batch_size:
				continue

			data, target = data.to(self.device), target.to(self.device)
			self.optimizer.zero_grad()
			out = self.model(data)

			if str(self.criterion) == 'CrossEntropyLoss()':
				loss = self.criterion(out, target)

				if self.use_consistency_loss:
					handle = None
					for layer_name, layer in self.layers:
						if handle is not None:
							handle.remove()
						handle = layer.register_forward_hook(self.get_hook())
						self.model(data)
						intermidiate_loss = self.consis_criterion(out, target, intermediate_features=self.train_features.value)
						loss += intermidiate_loss

					if handle is not None:
						handle.remove()
					gc.collect()


			elif str(self.criterion) == 'MSELoss()':
				loss = self.criterion(out, F.one_hot(target, num_classes=self.num_classes).float())

			loss.backward()
			self.optimizer.step()

			accuracy = torch.mean((torch.argmax(out, dim=1) == target).float()).item()

			pbar.update(1)

			if self.debug and batch_idx > 20:
				break
		pbar.close()

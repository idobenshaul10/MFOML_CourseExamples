from dataclasses import dataclass
import torch.nn as nn
import torch
from tqdm import tqdm
import wandb
import random
import numpy as np
from torchvision import datasets, transforms

torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
g = torch.Generator()
g.manual_seed(0)
def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2 ** 32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

class Features:
	pass


@dataclass
class Analyzer:
	conf: dict
	model: nn.Module
	layers: list
	num_classes: int
	device: torch.device
	criterion_summed: nn.Module
	train_loader: torch.utils.data.DataLoader = None
	test_loader: torch.utils.data.DataLoader = None
	handle: torch.utils.hooks.RemovableHandle = None
	debug: bool = True

	def __post_init__(self):
		transform = transforms.Compose([transforms.Pad((self.conf['padded_im_size'] - self.conf['im_size']) // 2),
										transforms.ToTensor(),
										transforms.Normalize(tuple(self.conf['dataset_mean']),
															 tuple(self.conf['dataset_std']))])
		if self.conf['dataset'] != 'STL10':
			train_dataset = eval(
				f'datasets.{self.conf["dataset"]}("../data", train=True, download=True, transform=transform)')
			test_dataset = eval(
				f'datasets.{self.conf["dataset"]}("../data", train=False, download=True, transform=transform)')

		else:
			train_dataset = eval(
				f'datasets.{self.conf["dataset"]}("../data", split="train", download=True, transform=transform)')
			test_dataset = eval(
				f'datasets.{self.conf["dataset"]}("../data", split="test", download=True, transform=transform)')

		self.train_loader = torch.utils.data.DataLoader(train_dataset,
					    batch_size=self.conf['batch_size'], shuffle=True,
					    worker_init_fn=seed_worker,
					    generator=g)
		if test_dataset is not None:
			self.test_loader = torch.utils.data.DataLoader(test_dataset,
						batch_size=self.conf['batch_size'], shuffle=True,
						worker_init_fn=seed_worker,
						generator=g)

		self.features = Features()

	def get_hook(self):
		def hook(model, input, output):
			self.features.value = input[0].clone()
		return hook

	def handle_layer_train(self, layer, layer_name, epoch, result:dict=None, is_first: bool=False):
		if self.handle is not None:
			self.handle.remove()
		self.handle = layer.register_forward_hook(self.get_hook())
		self.N = [0 for _ in range(self.num_classes)]
		self.mean = [0 for _ in range(self.num_classes)]

		loss = 0
		net_correct = 0
		NCC_match_net = 0

		for computation in ['Mean', 'Cov']:
			pbar = tqdm(total=len(self.train_loader), position=0, leave=True)
			for batch_idx, (data, target) in enumerate(self.train_loader, start=1):

				data, target = data.to(self.device), target.to(self.device)
				output = self.model(data)
				h = self.features.value.data.view(data.shape[0], -1)  # B CHW

				if is_first and computation == 'Mean':
					loss += self.criterion_summed(output, target).item()

				output = output.cpu()
				h = h.cpu()
				for c in range(self.num_classes):
					# features belonging to class c
					idxs = (target == c).nonzero(as_tuple=True)[0]

					if len(idxs) == 0:  # If no class-c in this batch
						continue

					h_c = h[idxs, :].cpu()  # B CHW
					if computation == 'Mean':
						self.mean[c] += torch.sum(h_c, dim=0)  #  CHW
						self.N[c] += h_c.shape[0]

					elif computation == 'Cov':
						net_pred = torch.argmax(output[idxs, :], dim=1)
						target = target.cpu()
						if is_first:
							net_correct += sum(net_pred == target[idxs]).item()

						NCC_scores = torch.stack([torch.norm(h_c[i, :] - self.M.T, dim=1) \
												  for i in range(h_c.shape[0])])

						NCC_pred = torch.argmin(NCC_scores, dim=1)
						NCC_match_net += sum(NCC_pred == net_pred).item()

				pbar.update(1)
				pbar.set_description(
					'Analysis {}\t'
					f'Layer {layer_name}\t'
					'Epoch: {} [{}/{} ({:.0f}%)]'.format(
						computation,
						epoch,
						batch_idx,
						len(self.train_loader),
						100. * batch_idx / len(self.train_loader)))

				if self.debug and batch_idx > 20:
					break

			pbar.close()

			if computation == 'Mean':
				for c in range(self.num_classes):
					self.mean[c] /= self.N[c]
					self.M = torch.stack(self.mean).T
				loss /= sum(self.N)

		if is_first:
			train_error = 1 - net_correct / sum(self.N)
			result.update({'train_error': train_error, "loss": loss, "train_accuracy": net_correct / sum(self.N)})
		result.update({f"NCC_mismatch_{layer_name}": 1 - NCC_match_net / sum(self.N)})

	def handle_layer_test(self, layer, layer_name, epoch, result:dict=None, is_first: bool=False):
		if self.handle is not None:
			self.handle.remove()
		self.handle = layer.register_forward_hook(self.get_hook())
		N = [0 for _ in range(self.num_classes)]
		mean = [0 for _ in range(self.num_classes)]

		net_correct = 0
		NCC_match_net = 0

		pbar = tqdm(total=len(self.test_loader), position=0, leave=True)
		for batch_idx, (data, target) in enumerate(self.test_loader, start=1):

			data, target = data.to(self.device), target.to(self.device)
			output = self.model(data)
			h = self.features.value.data.view(data.shape[0], -1)  # B CHW


			output = output.cpu()
			h = h.cpu()
			for c in range(self.num_classes):
				idxs = (target == c).nonzero(as_tuple=True)[0]

				if len(idxs) == 0:  # If no class-c in this batch
					continue

				h_c = h[idxs, :].cpu()  # B CHW
				net_pred = torch.argmax(output[idxs, :], dim=1)
				N[c] += h_c.shape[0]
				target = target.cpu()
				if is_first:
					net_correct += sum(net_pred == target[idxs]).item()

				NCC_scores = torch.stack([torch.norm(h_c[i, :] - self.M.T, dim=1) \
										  for i in range(h_c.shape[0])])

				NCC_pred = torch.argmin(NCC_scores, dim=1)
				NCC_match_net += sum(NCC_pred == net_pred).item()

			pbar.update(1)
			pbar.set_description(
				'Analysis {}\t'
				f'Layer {layer_name}\t'
				'Epoch: {} [{}/{} ({:.0f}%)]'.format(
					"test",
					epoch,
					batch_idx,
					len(self.train_loader),
					100. * batch_idx / len(self.train_loader)))

			if self.debug and batch_idx > 20:
				break

		pbar.close()
		if is_first:
			test_error = 1 - net_correct / sum(N)
			result.update({'test_error': test_error, "test_accuracy": net_correct / sum(N)})
		result.update({f"test_NCC_mismatch_{layer_name}": 1 - NCC_match_net / sum(N)})

	def handle_layers(self, epoch):
		result = {'epoch': epoch}
		for idx, (layer_name, layer) in enumerate(self.layers):
			self.handle_layer_train(layer, layer_name, epoch, result, is_first=(idx == 0))
			if self.test_loader is not None:
				self.handle_layer_test(layer, layer_name, epoch, result, is_first=(idx == 0))
			# break

		print(result)
		wandb.log(result)
		print("after logging to wandb")
		return result

	def analyze(self, epoch):
		handle = None
		self.model.eval()
		result = self.handle_layers(epoch)

		if handle is not None:
			handle.remove()
		return result

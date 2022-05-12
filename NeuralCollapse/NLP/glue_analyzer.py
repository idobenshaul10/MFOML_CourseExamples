import argparse
import gc
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
	AdamW,
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	DataCollatorWithPadding,
	PretrainedConfig,
	SchedulerType,
	default_data_collator,
	get_scheduler,
	set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
import sys
import torch.nn as nn

sys.path.append("/home/ido/projects/NeuralCollapse")
sys.path.append("/home/ido/projects/NeuralCollapse/NLP")
import torch
import wandb

logger = logging.getLogger(__name__)

@dataclass
class GlueAnazlyzer:

	model: nn.Module
	train_loader: torch.utils.data.DataLoader = None
	test_loader: torch.utils.data.DataLoader = None
	num_classes: int = 2


	def handle_layer_train(self, layer_idx, result: dict = None, is_first: bool = False):
		self.N = [0 for _ in range(self.num_classes)]
		self.mean = [0 for _ in range(self.num_classes)]

		loss = 0
		NCC_match_net = 0

		for computation in ['Mean', 'Cov']:
			print(f"IN {computation}, layer_idx:{layer_idx}!!")
			pbar = tqdm(total=len(self.train_loader), position=0, leave=True)
			for step, batch in enumerate(self.train_loader):
				batch = {k: v.cuda() for k, v in batch.items()}
				target = batch['labels'].cpu()
				outputs = self.model(**batch)
				# outputs = self.model._slow_forward(**batch)
				cur_loss = outputs.loss
				h = outputs.hidden_states[layer_idx].data.view(len(batch['labels']), -1).cpu()  # B CHW
				logits = outputs['logits'].cpu()
				del outputs

				if is_first and computation == 'Mean':
					loss += cur_loss.cpu().item()


				for c in range(self.num_classes):
					idxs = (batch['labels'].cpu() == c).nonzero(as_tuple=True)[0]

					if len(idxs) == 0:  # If no class-c in this batch
						continue

					h_c = h[idxs, :].cpu()  # B CHW
					if computation == 'Mean':
						try:
							self.mean[c] += torch.sum(h_c, dim=0)  #  CHW
							self.N[c] += h_c.shape[0]
						except Exception as e:
							print(e)


					elif computation == 'Cov':
						net_pred = torch.argmax(logits[idxs, :], dim=1)

						NCC_scores = torch.stack([torch.norm(h_c[i, :] - self.M.T, dim=1) \
												  for i in range(h_c.shape[0])])

						NCC_pred = torch.argmin(NCC_scores, dim=1)
						NCC_match_net += sum(NCC_pred == net_pred).item()

				gc.collect()
				pbar.update(1)


			if computation == 'Mean':
				for c in range(self.num_classes):
					self.mean[c] /= self.N[c]
					self.M = torch.stack(self.mean).T
				loss /= sum(self.N)
			pbar.close()

		self.model.zero_grad()
		if is_first:
			result.update({"loss": loss})
		result.update({f"NCC_mismatch_{layer_idx}": 1 - NCC_match_net / sum(self.N)})

	def handle_layer_test(self, layer_idx, result: dict = None):
		N = [0 for _ in range(self.num_classes)]

		NCC_match_net = 0

		pbar = tqdm(total=len(self.test_loader), position=0, leave=True)
		for step, batch in enumerate(self.test_loader):
			batch = {k: v.cuda() for k, v in batch.items()}
			target = batch['labels'].cpu()
			outputs = self.model(**batch)
			cur_loss = outputs.loss
			h = outputs.hidden_states[layer_idx].data.view(len(batch['labels']), -1).cpu()  # B CHW
			logits = outputs['logits'].cpu()
			del outputs


			for c in range(self.num_classes):
				idxs = (batch['labels'].cpu() == c).nonzero(as_tuple=True)[0]
				if len(idxs) == 0:  # If no class-c in this batch
					continue

				h_c = h[idxs, :].cpu()  # B CHW
				net_pred = torch.argmax(logits[idxs, :], dim=1)
				N[c] += h_c.shape[0]

				NCC_scores = torch.stack([torch.norm(h_c[i, :] - self.M.T, dim=1) \
										  for i in range(h_c.shape[0])])

				NCC_pred = torch.argmin(NCC_scores, dim=1)
				NCC_match_net += sum(NCC_pred == net_pred).item()

			gc.collect()
			pbar.update(1)

		pbar.close()
		result.update({f"test_NCC_mismatch_{layer_idx}": 1 - NCC_match_net / sum(N)})

	def handle_layers(self, result, state_dict):

		self.model.eval()
		for layer_idx in tqdm(range(0, 0)):
			self.handle_layer_train(layer_idx, result, is_first=(layer_idx == 0))
			if self.test_loader is not None:
				self.handle_layer_test(layer_idx, result)

		print(result)
		wandb.log(result)
		print("after logging to wandb")
		return self.model

	def analyze(self, result, state_dict):
		# print(f"in analyze, result:{result}")
		self.handle_layers(result, state_dict)
		return

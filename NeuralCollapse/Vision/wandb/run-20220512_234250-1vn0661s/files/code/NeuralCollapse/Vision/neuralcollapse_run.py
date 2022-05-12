import sys
import torch

from analysis import Analyzer
import gc
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from IPython import embed
import datetime
import pickle
import wandb
import random
from losses import ConsistancyLoss
from init_loader import init
import argparse

def main(conf, next_parameters):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	lr_decay = 0.1
	epoch_list = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101, 110, 121, 132, 144, 158, 172, 188, 206,
				  225, 245, 268, 293, 320, 350]

	print(f"next_parameters:{next_parameters}")

	conf, model, trainer, criterion_summed, device, num_classes, epochs, epochs_lr_decay, dataset = init(conf,
					use_consistency_loss=next_parameters['use_consistency_loss'], next_parameters=next_parameters)

	layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']
	eval_layers = []
	for layer_name in layer_names:
		layer = eval(f"model.{layer_name}")
		eval_layers.append((layer_name, layer))
	analyzer = Analyzer(conf, model, eval_layers, num_classes, device, criterion_summed)

	lr_scheduler = optim.lr_scheduler.MultiStepLR(trainer.optimizer,
												  milestones=epochs_lr_decay,
												  gamma=lr_decay)

	cur_epochs = []
	for epoch in range(1, epochs + 1):
		print(f"Starting epoch {epoch}")
		trainer.train(epoch)
		lr_scheduler.step()

		if epoch in epoch_list:
			cur_epochs.append(epoch)
			result = analyzer.analyze(epoch)


def run_main(next_parameters, config_path):
	dataset_config = pickle.load(open(config_path, "rb"))
	main(conf=dataset_config, next_parameters=next_parameters)

if __name__ == '__main__':
	alpha_const, layers_from_end, use_const = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'True'
	config_path = sys.argv[4]
	next_parameters = {'alpha_consis': alpha_const,
					   'num_layers_from_end': layers_from_end, 'use_consistency_loss': use_const}
	print(next_parameters)
	run_main(next_parameters, config_path)
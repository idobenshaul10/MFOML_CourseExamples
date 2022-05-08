import os
import sys

currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from sparsity_analyzer import SparsityAnalyzer

import numpy as np
from datetime import datetime
import torch
from torch import save
import torch.nn as nn
import argparse
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import importlib
from shutil import copyfile
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import wandb
import torch.optim as optim


# USAGE:  python .\train\train_mnist.py --output_path "results\" --batch_size 32 --epochs 100

def get_args():
	parser = argparse.ArgumentParser(description='train a model from enviorment')
	parser.add_argument('--output_path', help='output_path for checkpoints')
	parser.add_argument('--seed', default=0, type=int, help='seed')
	parser.add_argument('--lr', default=0.1, type=float, help='lr for train')
	parser.add_argument('--batch_size', default=128, type=int, help='batch_size for train/test')
	parser.add_argument('--epochs', default=501, type=int, help='num epochs for train')
	parser.add_argument('--env_name', type=str, default="cifar10_resnet_env")
	parser.add_argument('--save_epochs', action="store_true")
	parser.add_argument('--use_norms', action="store_true")
	args, _ = parser.parse_known_args()

	args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	return args


softmax = nn.Softmax(dim=1)


def train(train_loader, model, criterion, optimizer, device):
	model.train()
	running_loss = 0
	for X, y_true in tqdm(train_loader, total=len(train_loader)):
		optimizer.zero_grad()
		X = X.to(device)
		y_true = y_true.to(device)
		y_hat = model(X)
		loss = criterion(y_hat, y_true.long())
		running_loss += loss.item() * X.size(0)
		loss.backward()
		optimizer.step()

	epoch_loss = running_loss / len(train_loader.dataset)
	return model, optimizer, epoch_loss


def validate(valid_loader, model, criterion, device):
	model.eval()
	running_loss = 0

	for X, y_true in valid_loader:
		X = X.to(device)
		y_true = y_true.to(device).long()
		y_hat = model(X)
		loss = criterion(y_hat, y_true)
		running_loss += loss.item() * X.size(0)

	epoch_loss = running_loss / len(valid_loader.dataset)
	return model, epoch_loss


def get_accuracy(model, data_loader, device):
	correct_pred = 0
	n = 0
	with torch.no_grad():
		model.eval()
		for X, y_true in data_loader:
			X = X.to(device)
			y_true = y_true.to(device).long()
			logits = model(X)
			probs = softmax(logits)
			predicted_labels = torch.max(probs, 1)[1]
			n += y_true.size(0)
			correct_pred += (predicted_labels == y_true).sum()
	return correct_pred.float() / n


def save_epoch(output_path, epoch, model, train_acc, valid_acc, epoch_checkpoint=False):
	if not os.path.isdir(output_path):
		os.mkdir(output_path)
	if epoch_checkpoint:
		file_name = f"weights_{epoch}.h5"
	else:
		file_name = f"weights.best.h5"
	checkpoint_path = os.path.join(output_path, file_name)
	model_state_dict = model.state_dict()
	state_dict = OrderedDict()
	state_dict["epoch"] = epoch
	state_dict["checkpoint"] = model_state_dict
	state_dict["train_acc"] = train_acc
	state_dict["valid_acc"] = valid_acc
	save(state_dict, checkpoint_path)
	print("saved to {}".format(checkpoint_path))


def training_loop(model, criterion, optimizer, train_loader, valid_loader,
				  epochs, device, use_index, print_every=1, save_epochs=False):
	analyzer = SparsityAnalyzer(layers=layers, train_loader=train_loader, use_index=use_index)
	lr_decay = 0.1
	train_losses = []
	valid_losses = []
	best_val_acc = -1
	epochs_lr_decay = [25, 50, 75]
	lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
												  milestones=epochs_lr_decay,
												  gamma=lr_decay)

	train_acc = get_accuracy(model, train_loader, device=device)
	valid_acc = get_accuracy(model, valid_loader, device=device)

	# result = {'epoch': 0, 'train_metric': train_acc,
	# 		  'eval_metric': valid_acc}
	# analyzer.analyze(model, result)

	for epoch in range(0, epochs):
		model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
		train_losses.append(train_loss)
		lr_scheduler.step()

		with torch.no_grad():
			model, valid_loss = validate(valid_loader, model, criterion, device)
			valid_losses.append(valid_loss)

		if epoch % print_every == (print_every - 1):
			train_acc = get_accuracy(model, train_loader, device=device)
			valid_acc = get_accuracy(model, valid_loader, device=device)

			print(f'{datetime.now().time().replace(microsecond=0)} --- '
				  f'Epoch: {epoch}\t'
				  f'Train loss: {train_loss:.4f}\t'
				  f'Valid loss: {valid_loss:.4f}\t'
				  f'Train accuracy: {100 * train_acc:.2f}\t'
				  f'Valid accuracy: {100 * valid_acc:.2f}')

		if epoch == epochs - 1:
			if save_epochs:
				os.makedirs("checkpoints", exist_ok=True)
				save_epoch("checkpoints", epoch, model, train_acc, valid_acc, epoch_checkpoint=False)
			result = {'epoch': epoch, 'train_metric': train_acc,
					  'eval_metric': valid_acc, 'train_loss': train_loss, 'valid_loss': valid_loss}
			print(f"begining analysis, result:{result}")
			import pdb; pdb.set_trace()
			analyzer.analyze(model, result)

	return model, optimizer, (train_losses, valid_losses)


if __name__ == '__main__':
	args = get_args()
	torch.manual_seed(args.seed)
	wandb.init(project='SparsityForCourse', entity='ibenshaul', mode="disabled", tags=["Vision", "using_norm"])
	wandb.config.update(args)

	m = '.'.join(['environments', args.env_name])

	module = importlib.import_module(m)
	dict_input = vars(args)
	environment = eval(f"module.{args.env_name}()")

	model, train_dataset, test_dataset, layers = environment.load_environment(**dict_input)

	time_filename = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
	# import pdb;	pdb.set_trace()
	if args.output_path is not None:
		output_path = os.path.join(args.output_path, f"{args.env_name}_{time_filename}")
		if not os.path.isdir(output_path):
			os.mkdir(output_path)

		path = Path(__file__)
		model_path = os.path.join(path.parents[1], 'models', f"{type(model).__name__}.py")
		copyfile(model_path, os.path.join(output_path, "model.py"))
		pickle.dump(args, open(os.path.join(output_path, "args.p"), "wb"))

	print(f"Begining train, args:{args}")
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	criterion = nn.CrossEntropyLoss()

	train_loader = DataLoader(dataset=train_dataset,
							  batch_size=args.batch_size,
							  shuffle=True)

	test_loader = DataLoader(dataset=test_dataset,
							 batch_size=args.batch_size,
							 shuffle=False)

	model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, test_loader, args.epochs,
										args.device, use_index=not args.use_norms, save_epochs=args.save_epochs)

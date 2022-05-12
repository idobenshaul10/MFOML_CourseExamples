# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
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
sys.path.append("/home/ido/projects/NeuralCollapse")
# sys.path.append("/home/ido/projects/NeuralCollapse/NLP")
from NLP.glue_analyzer import GlueAnazlyzer
import torch
import wandb
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
torch.backends.cudnn.deterministic = True
import torch.distributed as dist
import torchvision.models as models
from torchvision import datasets as TV_datasets

import numpy as np
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

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
	"cola": ("sentence", None),
	"mnli": ("premise", "hypothesis"),
	"mrpc": ("sentence1", "sentence2"),
	"qnli": ("question", "sentence"),
	"qqp": ("question1", "question2"),
	"rte": ("sentence1", "sentence2"),
	"sst2": ("sentence", None),
	"stsb": ("sentence1", "sentence2"),
	"wnli": ("sentence1", "sentence2"),
}


def parse_args():
	parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
	parser.add_argument(
		"--task_name",
		type=str,
		default='sst2',
		help="The name of the glue task to train on.",
		choices=list(task_to_keys.keys()),
	)
	parser.add_argument(
		"--train_file", type=str, default=None, help="A csv or a json file containing the training data."
	)
	parser.add_argument(
		"--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
	)
	parser.add_argument(
		"--max_length",
		type=int,
		default=32,
		help=(
			"The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
			" sequences shorter will be padded if `--pad_to_max_lengh` is passed."
		),
	)
	parser.add_argument(
		"--pad_to_max_length",
		action="store_false",
		help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
	)
	parser.add_argument(
		"--use_consistency_loss",
		action="store_true",
		help=".",
	)
	parser.add_argument(
		"--model_name_or_path",
		type=str,
		default='bert-base-uncased',
		help="Path to pretrained model or model identifier from huggingface.co/models.",
		# required=True,
	)
	parser.add_argument(
		"--use_slow_tokenizer",
		action="store_true",
		help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
	)
	parser.add_argument(
		"--per_device_train_batch_size",
		type=int,
		default=8,
		help="Batch size (per device) for the training dataloader.",
	)
	parser.add_argument(
		"--per_device_eval_batch_size",
		type=int,
		default=8,
		help="Batch size (per device) for the evaluation dataloader.",
	)
	parser.add_argument(
		"--learning_rate",
		type=float,
		default=5e-5,
		help="Initial learning rate (after the potential warmup period) to use.",
	)
	parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
	parser.add_argument("--consis_param", type=float, default=1e-5, help="consistency alpha.")
	parser.add_argument("--first_layer_used", type=int, default=0, help="first layer used")
	parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
	parser.add_argument(
		"--max_train_steps",
		type=int,
		default=None,
		help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
	)
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--lr_scheduler_type",
		type=SchedulerType,
		default="linear",
		help="The scheduler type to use.",
		choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
	)
	parser.add_argument(
		"--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
	parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
	parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
	parser.add_argument(
		"--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
	)

	parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
	parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
	args = parser.parse_args()

	# Sanity checks
	if args.task_name is None and args.train_file is None and args.validation_file is None:
		raise ValueError("Need either a task name or a training/validation file.")
	else:
		if args.train_file is not None:
			extension = args.train_file.split(".")[-1]
			assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
		if args.validation_file is not None:
			extension = args.validation_file.split(".")[-1]
			assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

	if args.push_to_hub:
		assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

	return args


class GlueConsistencyLoss:
	def __init__(self, num_layers, num_classes, consis_param=1e-5, first_layer_used=4):
		self.C = num_classes
		self.alpha_coef = consis_param
		self.alpha = (1 / num_layers) * self.alpha_coef
		self.first_layer_used = first_layer_used
		try:
			wandb.config.update({"alpha": self.alpha_coef, "first_layer_used":self.first_layer_used})
		except:
			print("wandb not initiated")

	def __call__(self, outputs, target):
		mse_loss = 0.
		for layer_idx in range(self.first_layer_used, len(outputs.hidden_states) - 1):
			intermediate_features = outputs.hidden_states[layer_idx]
			for i in range(self.C):
				indices = torch.where(target == i)[0]
				cur_mse_loss = 0
				if len(indices) == 0:
					continue
				class_mean = torch.mean(intermediate_features[indices], dim=0)
				for j in range(len(indices)):
					cur_mse_loss += torch.sum((intermediate_features[indices[j]] - class_mean) ** 2)
				cur_mse_loss /= len(indices)
				mse_loss += cur_mse_loss

		mse_loss /= self.C
		return self.alpha * mse_loss


class train_features:
	pass


def hook(self, input, output):
	train_features.value = input[0].clone()


def main(params):
	args = parse_args()
	args.consis_param = params['consis_param']
	args.first_layer_used = params['first_layer_used']
	# args.task_name = params['task_name']
	print(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	wandb.init(project='<project_name>', entity='<user_entity>', mode="enabled",
			   tags=["NLP", args.task_name, args.model_name_or_path], reinit=True)
	wandb.config.update(args)

	accelerator = Accelerator()
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	logger.info(accelerator.state)

	logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
	if accelerator.is_local_main_process:
		datasets.utils.logging.set_verbosity_warning()
		transformers.utils.logging.set_verbosity_info()
	else:
		datasets.utils.logging.set_verbosity_error()
		transformers.utils.logging.set_verbosity_error()

	# If passed along, set the training seed now.
	if args.seed is not None:
		set_seed(args.seed)

	# Handle the repository creation
	if accelerator.is_main_process:
		if args.push_to_hub:
			if args.hub_model_id is None:
				repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
			else:
				repo_name = args.hub_model_id
			repo = Repository(args.output_dir, clone_from=repo_name)
		elif args.output_dir is not None:
			os.makedirs(args.output_dir, exist_ok=True)
	accelerator.wait_for_everyone()

	if args.task_name is not None:
		raw_datasets = load_dataset("glue", args.task_name)
	else:
		# Loading the dataset from local csv or json file.
		data_files = {}
		if args.train_file is not None:
			data_files["train"] = args.train_file
		if args.validation_file is not None:
			data_files["validation"] = args.validation_file
		extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
		raw_datasets = load_dataset(extension, data_files=data_files)

	# Labels
	if args.task_name is not None:
		is_regression = args.task_name == "stsb"
		if not is_regression:
			label_list = raw_datasets["train"].features["label"].names
			num_labels = len(label_list)
		else:
			num_labels = 1
	else:
		# Trying to have good defaults here, don't hesitate to tweak to your needs.
		is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
		if is_regression:
			num_labels = 1
		else:
			# A useful fast method:
			# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
			label_list = raw_datasets["train"].unique("label")
			label_list.sort()  # Let's sort it for determinism
			num_labels = len(label_list)

	# Load pretrained model and tokenizer
	#
	# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.
	config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name,
										output_hidden_states=True)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
	model = AutoModelForSequenceClassification.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
	)

	if args.use_consistency_loss:
		print(f"Inited trainer, using consistency loss")
		consis_criterion = GlueConsistencyLoss(num_layers=12, num_classes=2,
											   consis_param=args.consis_param, first_layer_used=args.first_layer_used)



	# Preprocessing the datasets
	if args.task_name is not None:
		sentence1_key, sentence2_key = task_to_keys[args.task_name]
	else:
		# Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
		non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
		if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
			sentence1_key, sentence2_key = "sentence1", "sentence2"
		else:
			if len(non_label_column_names) >= 2:
				sentence1_key, sentence2_key = non_label_column_names[:2]
			else:
				sentence1_key, sentence2_key = non_label_column_names[0], None

	# Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = None
	if (
			model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
			and args.task_name is not None
			and not is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
			logger.info(
				f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
				"Using it!"
			)
			label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
		else:
			logger.warning(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
				"\nIgnoring the model labels as a result.",
			)
	elif args.task_name is None:
		label_to_id = {v: i for i, v in enumerate(label_list)}

	if label_to_id is not None:
		model.config.label2id = label_to_id
		model.config.id2label = {id: label for label, id in config.label2id.items()}
	elif args.task_name is not None and not is_regression:
		model.config.label2id = {l: i for i, l in enumerate(label_list)}
		model.config.id2label = {id: label for label, id in config.label2id.items()}

	padding = "max_length" if args.pad_to_max_length else False

	def preprocess_function(examples):
		# Tokenize the texts
		texts = (
			(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		)
		result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

		if "label" in examples:
			if label_to_id is not None:
				# Map labels to IDs (not necessary for GLUE tasks)
				result["labels"] = [label_to_id[l] for l in examples["label"]]
			else:
				# In all cases, rename the column to labels because the model will expect that.
				result["labels"] = examples["label"]
		return result

	with accelerator.main_process_first():
		processed_datasets = raw_datasets.map(
			preprocess_function,
			batched=True,
			remove_columns=raw_datasets["train"].column_names,
			desc="Running tokenizer on dataset",
		)

	train_dataset = processed_datasets["train"]
	eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

	# Log a few random samples from the training set:
	for index in random.sample(range(len(train_dataset)), 3):
		logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

	if args.pad_to_max_length:
		data_collator = default_data_collator
	else:
		data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

	train_dataloader = DataLoader(
		train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
	)
	eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator,
								 shuffle=False, batch_size=args.per_device_eval_batch_size)

	train_dataloader_analysis = DataLoader(
		train_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,
								worker_init_fn=seed_worker, generator=g)

	eval_dataloader_analysis = DataLoader(eval_dataset, collate_fn=data_collator,
								shuffle=False, batch_size=args.per_device_eval_batch_size,
								worker_init_fn=seed_worker, generator=g)

	train_dataset_3 = TV_datasets.MNIST("../data", train=True, download=True)
	# fake_loader = torch.utils.data.DataLoader(train_dataset_3,
	# 										   batch_size=128, shuffle=False, worker_init_fn=seed_worker,
	# 										   generator=g)

	# Optimizer
	# Split weights in two groups, one with weight decay and the other not.
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{
			"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
			"weight_decay": 0.0,
		},
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
	model = model.cuda()

	analyzer = GlueAnazlyzer(model=model, train_loader=train_dataloader_analysis, test_loader=eval_dataloader_analysis)

	# Scheduler and math around the number of training steps.
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
	if args.max_train_steps is None:
		args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
	else:
		args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

	lr_scheduler = get_scheduler(
		name=args.lr_scheduler_type,
		optimizer=optimizer,
		num_warmup_steps=args.num_warmup_steps,
		num_training_steps=args.max_train_steps,
	)

	# Get the metric function
	if args.task_name is not None:
		metric = load_metric("glue", args.task_name)
	else:
		metric = load_metric("accuracy")

	# Train!
	total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
	logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {args.max_train_steps}")
	progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
	completed_steps = 0

	for epoch in range(args.num_train_epochs):
		model.train()
		for step, batch in enumerate(train_dataloader):
			batch = {k: v.cuda() for k, v in batch.items()}
			outputs = model(**batch)
			loss = outputs.loss

			if args.use_consistency_loss:
				intermidiate_loss = consis_criterion(outputs, batch['labels'])
				loss += intermidiate_loss


			loss = loss / args.gradient_accumulation_steps
			accelerator.backward(loss)
			if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()
				progress_bar.update(1)
				completed_steps += 1

			if completed_steps >= args.max_train_steps:
				break

		#EVALUATE
		model.eval()
		with torch.no_grad():
			preds_total, labels_total = [], []
			for step, batch in enumerate(train_dataloader_analysis):
				batch = {k: v.cuda() for k, v in batch.items()}
				outputs = model(**batch)
				predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
				# preds_total.extend(predictions.cpu().tolist())
				# labels_total.extend(batch['labels'].cpu().tolist())

				metric.add_batch(
					predictions=predictions,
					references=batch["labels"]
				)

			cur_metric = metric.compute()
			key = list(cur_metric.keys())[0]
			train_metric = cur_metric[key]
			print(f"key is {key}")

			preds_total, labels_total = [], []
			for step, batch in enumerate(eval_dataloader_analysis):
				batch = {k: v.cuda() for k, v in batch.items()}
				outputs = model(**batch)
				predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()

				metric.add_batch(
					predictions=predictions,
					references=batch["labels"]
				)
			cur_res = metric.compute()
			eval_metric = cur_res[key]
			# eval_metric = (np.array(preds_total) == np.array(labels_total)).sum() / len(preds_total)
			print(eval_metric)

		print(f"epoch {epoch}: train:{train_metric} eval: {eval_metric}")
		result = {'epoch': epoch, 'train_metric': train_metric, 'eval_metric': eval_metric}
		print(f"begining analysis, result:{result}")

		analyzer.analyze(result, model.state_dict())

	return eval_metric

if __name__ == "__main__":
	wandb.login(key="<login_id>")
	next_parameters = {'consis_param': 0.00004132464903683243, 'first_layer_used': 10}
	main(next_parameters)

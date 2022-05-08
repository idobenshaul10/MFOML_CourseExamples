import argparse
import time
import numpy as np
import importlib
from SparsityProbe.SparsityProbe import SparsityProbe
import argparse
import logging
import math
import os
import random
from pathlib import Path
import torch
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
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
import wandb

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

models = {
	'bert-base-cased': transformers.models.bert.modeling_bert.BertLayer,
	# 'microsoft/deberta-base': ,
	# 'squeezebert/squeezebert-uncased',
	'roberta-base': transformers.models.roberta.modeling_roberta.RobertaLayer,
	'xlnet-base-cased': transformers.models.xlnet.modeling_xlnet.XLNetLayer,
	'distilbert-base-cased': transformers.models.distilbert.modeling_distilbert.TransformerBlock,
	# 'albert-base-v1': transformers.models.albert.modeling_albert.AlbertLayer,
	# 'albert-base-v2': transformers.models.albert.modeling_albert.AlbertLayer,
}


def parse_args():
	parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
	parser.add_argument('--trees', default=5, type=int, help='Number of trees in the forest.')
	parser.add_argument('--depth', default=15, type=int, help='Maximum depth of each tree.Use 0 for unlimited depth.')
	# parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--epsilon_1', type=float, default=0.1)
	parser.add_argument('--apply_dim_reduction', action='store_true', default=False)

	parser.add_argument(
		"--task_name",
		type=str,
		default=None,
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
		"--model_name_or_path",
		type=str,
		help="Path to pretrained model or model identifier from huggingface.co/models.",
		required=True,
	)
	parser.add_argument(
		"--checkpoint_path",
		type=str,
		help="Path to pretrained model or model identifier from huggingface.co/models.",
		required=True,
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
	parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
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
	parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
	parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
	parser.add_argument(
		"--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
	)
	parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
	args = parser.parse_args()
	args.epsilon_2 = 4 * args.epsilon_1

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


def main():
	args = parse_args()
	wandb.init(project='SparsityForCourse', entity='ibenshaul', mode="online", tags=["NLP", "using_norm"])
	wandb.config.update(args)
	# Make one log on every process with the configuration for debugging.
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)

	# If passed along, set the training seed now.
	if args.seed is not None:
		set_seed(args.seed)

	if args.task_name is not None:
		# Downloading and loading a dataset from the hub.
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
	# See more about loading any type of standard or custom dataset at
	# https://huggingface.co/docs/datasets/loading_datasets.html.

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
	config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
	model = AutoModelForSequenceClassification.from_pretrained(
		args.model_name_or_path,
		from_tf=bool(".ckpt" in args.model_name_or_path),
		config=config,
	)

	print(f"loading model from {args.checkpoint_path}")
	state_dict = torch.load(args.checkpoint_path, map_location='cpu')
	model.load_state_dict(state_dict)
	print(f"MODEL LOADED")

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

	processed_datasets = raw_datasets.map(
		preprocess_function,
		batched=True,
		remove_columns=raw_datasets["train"].column_names,
		desc="Running tokenizer on dataset",
	)

	train_dataset = processed_datasets["train"]
	eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
	# eval_dataset = processed_datasets["test"]

	# Log a few random samples from the training set:
	for index in random.sample(range(len(train_dataset)), 3):
		logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

	# DataLoaders creation:
	if args.pad_to_max_length:
		# If padding was already done ot max length, we use the default data collator that will just convert everything
		# to tensors.
		data_collator = default_data_collator
	else:
		# Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
		# the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
		# of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
		data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(None))

	train_dataloader = DataLoader(
		train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
	)
	eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

	if args.task_name is not None:
		metric = load_metric("glue", args.task_name)
	else:
		metric = load_metric("accuracy")

	# ****************************************************************************
	# ### UNCOMMENT IF YOU WANT TO RUN EVALUATION ON THE TEST SET
	for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
		outputs = model(**batch)
		predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
		metric.add_batch(
			predictions=predictions,
			references=batch["labels"],
		)

	eval_metric = metric.compute()
	logger.info(f"test_metric: {eval_metric}")
	wandb.log({'eval_metric': eval_metric})
	# ****************************************************************************

	for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
		outputs = model(**batch)
		predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
		metric.add_batch(
			predictions=predictions,
			references=batch["labels"],
		)

	train_metric = metric.compute()
	logger.info(f"train_metric: {train_metric}")
	wandb.log({'train_metric': train_metric})
	# ****************************************************************************
	layers = [k for name, k in model.named_modules() if type(k) == models[args.model_name_or_path]]

	# cache_folder = r"/media/ido/Transcend/feature_cache/NLP/SST2"
	# Path(cache_folder).mkdir(parents=True, exist_ok=True)
	if torch.cuda.is_available():
		model = model.cuda()

	probe = SparsityProbe(train_dataloader, model, apply_dim_reduction=False, epsilon_1=args.epsilon_1,
						  epsilon_2=args.epsilon_2, n_trees=args.trees, depth=args.depth, n_state=args.seed,
						  layers=layers, compute_using_index=False)  # , layer_feature_cache_folder=cache_folder)

	# for layer in tqdm(probe.model_handler.layers[-1:]):
	for layer in tqdm(probe.model_handler.layers):
		start_time = time.time()
		alpha_score, alphas = probe.run_smoothness_on_layer(layer, text=f"{args.model_name_or_path}")
		layer_name = layer._get_name()
		print(f"alpha_score for {layer_name} is {alpha_score}, time:{time.time() - start_time}")
		wandb.log({"layer": layer_name, "alpha": alpha_score, "alphas_std": alphas.std()})


if __name__ == "__main__":
	main()

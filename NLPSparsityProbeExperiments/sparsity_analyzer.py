from dataclasses import dataclass
import torch
from SparsityProbe.SparsityProbe import SparsityProbe
from tqdm import tqdm
import time
import wandb

@dataclass
class SparsityAnalyzer:
	layers: list
	train_loader: torch.utils.data.DataLoader
	test_loader: torch.utils.data.DataLoader = None
	epsilon_1: float = 0.1
	epsilon_2: float = 0.4
	trees: int = 5
	depth: int = 15
	seed: int = 0
	use_index: bool = False

	def __post_init__(self):
		try:
			wandb.config.update({"use_index": self.use_index})
		except:
			print("wandb not initialized")

	def analyze(self, model, result):
		probe = SparsityProbe(self.train_loader, model, apply_dim_reduction=False, epsilon_1=self.epsilon_1,
									  epsilon_2=self.epsilon_2, n_trees=self.trees, depth=self.depth, n_state=self.seed,
									  layers=self.layers, compute_using_index=self.use_index)

		# for layer in tqdm(probe.model_handler.layers[-1:]):
		for layer_idx, layer in tqdm(enumerate(probe.model_handler.layers), total=len(probe.model_handler.layers)):
			start_time = time.time()
			alpha_score, alphas = probe.run_smoothness_on_layer(layer, text=f"")
			layer_name = layer._get_name()
			print(f"alpha_score for {layer_name}_{layer_idx} is {alpha_score}, time:{time.time() - start_time}")
			result.update({f"train_SparsityNorm_{layer_idx}": alpha_score})
			# wandb.log({"layer": layer_name, "alpha": alpha_score, "alphas_std": alphas.std()})
		wandb.log(result)
		print("after logging to wandb")
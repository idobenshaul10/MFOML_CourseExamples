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
	seed: int = 1079
	use_index: bool = True


	def analyze(self, model, result):
		compute_using_index = True
		probe = SparsityProbe(self.train_loader, model, apply_dim_reduction=True, epsilon_1=self.epsilon_1,
							epsilon_2=self.epsilon_2, n_trees=self.trees, depth=self.depth, n_state=self.seed,
							layers=self.layers, compute_using_index=self.use_index)
		wandb.log({'compute_using_index': compute_using_index})
		dataset_len = len(self.train_loader.dataset)
		data = self.train_loader.dataset.data.reshape(dataset_len, -1)
		try:
			data = data.numpy()
		except:
			print("data is already in np format")

		start_time = time.time()
		alpha_score, alphas = probe.run_smoothness_on_features(features=data)
		print(f"alpha_score for input_layer is {alpha_score}, time:{time.time() - start_time}")
		result.update({f"Alpha_0": alpha_score})


		for layer_idx, layer in tqdm(enumerate(probe.model_handler.layers), total=len(probe.model_handler.layers)):
			start_time = time.time()
			alpha_score, alphas = probe.run_smoothness_on_layer(layer, text=f"")
			layer_name = layer._get_name()
			print(f"alpha_score for {layer_name} is {alpha_score}, time:{time.time() - start_time}")
			result.update({f"Alpha_{layer_idx+1}": alpha_score})
		wandb.log(result)
		print("after logging to wandb")

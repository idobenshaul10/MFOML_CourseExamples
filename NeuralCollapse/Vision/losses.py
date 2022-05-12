import torch
import torch.nn as nn
import random
import numpy as np
import wandb

class ConsistancyLoss:
    def __init__(self, num_layers, num_classes, alpha_coef=5e-6):
        self.C = num_classes
        self.alpha_coef = alpha_coef
        self.alpha = (1/num_layers) * self.alpha_coef
        try:
            wandb.config.update({"alpha": self.alpha_coef})
        except:
            print("wandb not initiated")

    def __call__(self, out, target, intermediate_features):
        mse_loss = 0.
        for i in range(0, self.C):
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

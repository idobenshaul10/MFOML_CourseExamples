import glob
import json

from environments.base_environment import *
from torchvision import datasets, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import wandb
import os
from models.vit_ImageClassifier import ImageClassifier, ViTFeatureExtractorTransforms

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        # self.inp = torch.stack(list(transposed_data[0]), 0)
        self.tgt = torch.tensor(transposed_data[1])

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return {'pixel_values': self.inp, 'labels': self.tgt}


def my_collate(batch):
    return SimpleCustomBatch(batch)


class cifar10_vit(BaseEnvironment):
    def __init__(self, **kwargs):
        super().__init__()
        self.model_paths = json.load(open("models/model_paths.json"))
        self.model_name = 'google/vit-base-patch16-224-in21k'
        self.base_path = r"/home/ido/projects/huggingface-vit-finetune/lightning_logs/"

        checkpoints_folder_path = os.path.join(self.base_path, self.model_paths[self.model_name], "checkpoints")
        self.checkpoint_path = glob.glob(os.path.join(checkpoints_folder_path, "*.ckpt"))[0]


        # self.model_name = 'facebook/deit-tiny-patch16-224'
        # self.model_name = 'facebook/dino-vits8'

    def get_dataset(self):
        dataset = datasets.CIFAR10(root=r'cifar10',
                                   train=True,
                                   transform=self.get_eval_transform(),
                                   download=True)
        return dataset

    def get_test_dataset(self):
        dataset = datasets.CIFAR10(root=r'cifar10',
                                   train=False,
                                   transform=self.get_eval_transform(),
                                   download=True)
        return dataset

    def get_eval_transform(self):
        transform = ViTFeatureExtractorTransforms(self.model_name)
        # feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        # transform = transforms.Compose([
        #     transforms.Resize((feature_extractor.size, feature_extractor.size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=feature_extractor.image_mean,
        #                          std=feature_extractor.image_std)
        # ])
        return transform

    def get_layers(self, model):
        layers = []
        for name, module in model.named_modules():
            if ("model.vit.encoder.layer." in name and len(name) == len("model.vit.encoder.layer.0")) or name == "vit.embeddings":
                layers.append(module)
                print(name)
        return layers

    def get_model(self, **kwargs):
        if 'checkpoint_path' in kwargs:
            self.checkpoint_path = kwargs['checkpoint_path']

        wandb.config.update({"checkpoint_path": self.checkpoint_path}, allow_val_change=True)

        epoch = self.checkpoint_path.split('/')[-1].split('=')[2].split('-')[0]
        wandb.config.update({"epoch": epoch}, allow_val_change=True)

        # model = ImageClassifier.load_from_checkpoint(self.checkpoint_path)
        model = ImageClassifier(self.model_name)
        model = model.load_from_checkpoint(self.checkpoint_path)# load_state_dict(self.checkpoint_path)
        model.num_labels = 10
        wandb.config.update({"model": self.model_name})
        return model

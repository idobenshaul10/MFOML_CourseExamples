import torch
from dataclasses import dataclass
import gc

@dataclass
class BaseEnvironment:
    use_cuda: bool = False

    def __post_init__(self):
        self.use_cuda = torch.cuda.is_available()

    def load_environment(self, **kwargs):
        self.train_dataset = self.get_dataset()
        test_dataset = None
        try:
            test_dataset = self.get_test_dataset()
        except:
            print("test_dataset not available!")
        model = None
        model_cpu = self.get_model(**kwargs)
        if torch.cuda.is_available():
            model = model_cpu.cuda()

        layers = self.get_layers(model)
        # del model_cpu
        # del model
        # gc.collect()
        # model = None

        return model, self.train_dataset, test_dataset, layers

    def get_test_dataset(self):
        pass

    def get_layers(self, model):
        pass

    def get_dataset(self):
        pass

    def get_eval_transform(self):
        pass

    def get_model(self):
        pass

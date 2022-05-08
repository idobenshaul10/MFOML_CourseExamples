import argparse
import torch
import time
import numpy as np
from tqdm import tqdm
import importlib
from SparsityProbe.SparsityProbe import SparsityProbe
from torch.utils.data import Sampler
import pytorch_lightning as pl
import wandb

wandb.init(project='ImageNet_SparsityProbe', entity='ibenshaul', mode="online")
# wandb.init(project='ImageNet_SparsityProbe', entity='ibenshaul', mode="disabled")

def get_args():
    parser = argparse.ArgumentParser(description='Network Smoothness Script')
    parser.add_argument('--trees', default=3, type=int, help='Number of trees in the forest.')
    parser.add_argument('--depth', default=15, type=int, help='Maximum depth of each tree.Use 0 for unlimited depth.')
    parser.add_argument('--seed', type=int, default=2, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--env_name', type=str, default="mnist_1d_env")
    parser.add_argument('--checkpoints_folder', type=str, default=None)
    parser.add_argument('--epsilon_1', type=float, default=0.1)
    parser.add_argument('--output_folder', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--apply_dim_reduction', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--top_1_err', type=float, default=None)
    parser.add_argument('--top_5_err', type=float, default=None)

    parsed_args = parser.parse_args()
    parsed_args.epsilon_2 = 4 * parsed_args.epsilon_1
    return parsed_args


def init_params(args=None):
    if args is None:
        args = get_args()

    args.use_cuda = torch.cuda.is_available()
    print(args)
    wandb.config.update(args)

    m = '.'.join(['environments', args.env_name])
    module = importlib.import_module(m)

    environment = eval(f"module.{args.env_name}()")

    loaded_model, train_dataset, test_dataset, layers = environment.load_environment(**vars(args))
    # loaded_model = environment.get_model()
    # if torch.cuda.is_available():
    #     loaded_model = loaded_model.cuda()

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True,
                                               # collate_fn=my_collate
                                               )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size, shuffle=False,
                                              pin_memory=True,
                                              # collate_fn=my_collate
                                              )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args, loaded_model, train_dataset, test_dataset, train_loader, test_loader, layers


if __name__ == '__main__':
    args, model, dataset, test_dataset, data_loader, test_loader, layers = init_params()

    # For test metrics
    # **********************************************************************************
    if test_dataset is not None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer = pl.Trainer(gpus=1, max_epochs=1, precision=16, limit_train_batches=0)
        trainer.fit(model, data_loader, test_loader)

        result = trainer.test(model, test_loader)[0]
        wandb.log(result)

    probe = SparsityProbe(data_loader, model, apply_dim_reduction=args.apply_dim_reduction, epsilon_1=args.epsilon_1,
                          epsilon_2=args.epsilon_2, n_trees=args.trees, depth=args.depth, n_state=args.seed,
                          layers=layers)

    # for layer in tqdm(probe.model_handler.layers[-1:]):
    for layer in tqdm(probe.model_handler.layers[-2:]):
        start_time = time.time()
        alpha_score, alphas = probe.run_smoothness_on_layer(layer)
        layer_name = layer._get_name()
        print(f"alpha_score for {layer_name} is {alpha_score}, time:{time.time()-start_time}")
        wandb.log({"layer": layer_name, "alpha": alpha_score, "alphas_std": alphas.std()})

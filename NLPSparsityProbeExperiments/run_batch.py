import subprocess
from glob import glob
import os
from tqdm import tqdm
from subprocess import Popen
import numpy as np
import transformers
from eval import models


task = 'mrpc'

print("TRAINING MODEL!")
for model_name, _ in tqdm(models.items()):
    print(f"RUNNING {model_name}")
    command = f"python transformers/examples/pytorch/text-classification/run_glue.py  --model_name_or_path {model_name}  --task_name {task} --do_train --do_eval --max_seq_length 128   --per_gpu_train_batch_size 32  --learning_rate 2e-5   --num_train_epochs 10 --output_dir checkpoints/{task}/{model_name} "
    print(command)
    p = Popen(
        [
            command
        ],
        shell=True)
    p.wait()

print("MOVING TO EVAL!")
for model_name, _ in tqdm(models.items()):
    for seed in tqdm(np.arange(1, 2)):
        print(f"RUNNING {model_name}")
        command = f"python eval.py --model_name_or_path {model_name}  --task_name {task}  --learning_rate 2e-5   " \
                  f"--num_train_epochs 3   --output_dir /tmp/$TASK_NAME/ --checkpoint_path " \
                  f"checkpoints/{task}/{model_name}/pytorch_model.bin --per_device_train_batch_size 32 --seed {seed}"
        print(command)
        p = Popen(
            [
                command
            ],
            shell=True)
        p.wait()
# #
# #
# #
# #
# #
# #
# #
# #
# #
#
#
#
#
#
#
#






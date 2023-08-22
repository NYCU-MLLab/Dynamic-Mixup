import logging

import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

import json
import argparse

from transformers import AutoTokenizer

from model import ContrastNet

from paraphrase.utils.data import FewShotDataset

from utils.data import get_json_data, FewShotDataLoader
from utils.python import now, set_seeds
import random
import collections
import os
from typing import List, Dict, Callable, Union

from tensorboardX import SummaryWriter
import numpy as np
import warnings

from soft_prompt import SoftEmbedding
import time
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from transformers import (
    AdamW,
    get_scheduler
)
from transformers.optimization import get_linear_schedule_with_warmup

from torch.optim.lr_scheduler import ReduceLROnPlateau

start = time.time()
print(start)
initial_lambda = True
if(initial_lambda == True):
    l = np.random.beta(0.8, 0.8)
    if l<=0.5:
        l = 0.9
    initial_lambda = False

def run_fsintent(
        # Compulsory!
        data_path: str,
        train_labels_path: str,

        # Few-shot Stuff
        n_support: int,
        n_query: int,
        n_classes: int,
        model_name_or_path: str,
        super_tau: float = 1.0,
        lr: float = 1e-6,
        mixup_lr: float = 4e-5,
        metric: str = "euclidean",
        logger: object = None,
        super_weight: float = 0.7,
        max_len: int = 64,
        mixup_ratio: float = 0.9,
        k: int = 1,

        # Path training data ONLY (optional)
        train_path: str = None,

        # Validation & test
        valid_labels_path: str = None,
        test_labels_path: str = None,
        evaluate_every: int = 100,
        n_test_episodes: int = 1000,

        # Logging & Saving
        output_path: str = f'runs/{now()}',
        checkpoint_path: str = f'checkpoint/{now()}.pt',
        log_every: int = 10,

        # Training stuff
        max_iter: int = 10000,
        early_stop: int = None,
):
    if output_path:
        if os.path.exists(output_path) and len(os.listdir(output_path)):
            raise FileExistsError(f"Output path {output_path} already exists. Exiting.")

    # --------------------
    # Creating Log Writers
    # --------------------
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, "logs/train"))
    train_writer: SummaryWriter = SummaryWriter(logdir=os.path.join(output_path, "logs/train"), flush_secs=1, max_queue=1)
    valid_writer: SummaryWriter = None
    test_writer: SummaryWriter = None
    log_dict = dict(train=list())

    # ----------
    # Load model
    # ----------
   
    fsinet: ContrastNet = ContrastNet(config_name_or_path=model_name_or_path, metric=metric, max_len=max_len, super_tau=super_tau)
    for n, p in fsinet.named_parameters():
        print(n)
    
    
    optimizer_grouped_parameters = [
    {
        "params": [p for n, p in fsinet.named_parameters() if n == "encoder.embeddings.word_embeddings.soft_prompt.weight"],
        "weight_decay": 0.1,
    }
    ]
    mixup_ratio = torch.tensor(mixup_ratio, requires_grad=True)
    optimizer_parameters_lambda = [
    {
        "params": [mixup_ratio],
        "weight_decay": 0.1,
    }
    ]
    
    # optimizer = AdamW(fsinet.parameters(), lr=lr)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    optimizer_lambda = torch.optim.SGD(optimizer_parameters_lambda, lr=mixup_lr)
    logger.info(torch.cuda.list_gpu_processes())

    # ------------------
    # Load Train Dataset
    # ------------------
    train_dataset = FewShotDataset(
        data_path=train_path if train_path else data_path,
        labels_path=train_labels_path,
        n_classes=n_classes,
        n_support=n_support,
        n_query=n_query,
    )

    logger.info(f"Train dataset has {len(train_dataset)} items")

    # ---------
    # Load data
    # ---------
    logger.info(f"train labels: {train_dataset.data.keys()}")
    valid_dataset: FewShotDataset = None
    if valid_labels_path:
        os.makedirs(os.path.join(output_path, "logs/valid"))
        valid_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/valid"), flush_secs=1, max_queue=1)
        log_dict["valid"] = list()
        valid_dataset = FewShotDataset(data_path=data_path, labels_path=valid_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"valid labels: {valid_dataset.data.keys()}")
        assert len(set(valid_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    test_dataset: FewShotDataset = None
    if test_labels_path:
        os.makedirs(os.path.join(output_path, "logs/test"))
        test_writer = SummaryWriter(logdir=os.path.join(output_path, "logs/test"), flush_secs=1, max_queue=1)
        log_dict["test"] = list()
        test_dataset = FewShotDataset(data_path=data_path, labels_path=test_labels_path, n_classes=n_classes, n_support=n_support, n_query=n_query)
        logger.info(f"test labels: {test_dataset.data.keys()}")
        assert len(set(test_dataset.data.keys()) & set(train_dataset.data.keys())) == 0

    train_metrics = collections.defaultdict(list)
    n_eval_since_last_best = 0
    best_valid_acc = 0.0
    best_test_acc = 0.0
    best_valid_dict = None
    best_test_dict = None
    mixup_ratio_list=[]

    for step in range(max_iter):

        episode = train_dataset.get_episode()

        supervised_loss_share = super_weight*(1. - step/max_iter)
        

        if(mixup_ratio<=0.5):
            mixup_ratio = torch.tensor(0.51, requires_grad=True)
        if(mixup_ratio>=1):
            mixup_ratio = torch.tensor(0.99, requires_grad=True)
        mixup_ratio_list.append(mixup_ratio.item())
        loss, loss_dict = fsinet.train_step(optimizer=optimizer, optimizer_lambda = optimizer_lambda, l = mixup_ratio, k=k, episode=episode, supervised_loss_share=supervised_loss_share)

        for key, value in loss_dict["metrics"].items():
            train_metrics[key].append(value)

        # Logging
        if (step + 1) % log_every == 0:
            # scheduler.step(loss)
            for key, value in train_metrics.items():
                train_writer.add_scalar(tag=key, scalar_value=np.mean(value), global_step=step)
            logger.info(f"train | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in train_metrics.items()]))
            log_dict["train"].append({
                "metrics": [
                    {
                        "tag": key,
                        "value": np.mean(value)
                    }
                    for key, value in train_metrics.items()
                ],
                "global_step": step
            })

            train_metrics = collections.defaultdict(list)
            
        

        if valid_labels_path or test_labels_path:
            if (step + 1) % evaluate_every == 0:
                is_best = False
                for labels_path, writer, set_type, set_dataset in zip(
                        [valid_labels_path, test_labels_path],
                        [valid_writer, test_writer],
                        ["valid", "test"],
                        [valid_dataset, test_dataset]
                ):
                    if set_dataset:
                        set_results = fsinet.test_step(
                            l = mixup_ratio,
                            k=k,
                            dataset=set_dataset,
                            n_episodes=n_test_episodes
                            # z=z
                        )


                        for key, val in set_results.items():
                            writer.add_scalar(tag=key, scalar_value=val, global_step=step)
                        log_dict[set_type].append({
                            "metrics": [
                                {
                                    "tag": key,
                                    "value": val
                                }
                                for key, val in set_results.items()
                            ],
                            "global_step": step
                        })

                        logger.info(f"{set_type} | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in set_results.items()]))
                        if set_type == "valid":
                            if set_results["acc"] >= best_valid_acc:
                                best_valid_acc = set_results["acc"]
                                best_valid_dict = set_results
                                is_best = True
                                n_eval_since_last_best = 0
                                logger.info(f"Better eval results!")
                                torch.save(fsinet, checkpoint_path)
                                # fsinet.save_pretrained(checkpoint_path)
                                # torch.save(fsinet.state_dict(), checkpoint_path)
                            else:
                                n_eval_since_last_best += 1
                                logger.info(f"Worse eval results ({n_eval_since_last_best}/{early_stop})")
                        else:
                            if is_best:
                                best_test_dict = set_results

                if early_stop and n_eval_since_last_best >= early_stop:
                    logger.warning(f"Early-stopping.")
                    logger.info(f"Best eval results: ")
                    logger.info(f"valid | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_valid_dict.items()]))
                    logger.info(f"test | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_test_dict.items()]))
                    end = time.time()
                    logger.info(end-start)
                    break

    mixup_ratio_list1 = [round(i, 5) for i in mixup_ratio_list]
    plt.title('Mixup ratio')
    plt.xlabel('Iterations')
    plt.ylabel('lambda')
    plt.plot(mixup_ratio_list1)
    plt.legend()
    plt.savefig('mixup_ratio1e-5_1.png')

    logger.info(f"Best eval results: ")
    logger.info(f"valid | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_valid_dict.items()]))
    logger.info(f"test | " + " | ".join([f"{key}:{np.mean(value):.4f}" for key, value in best_test_dict.items()]))
    end = time.time()
    logger.info(end-start)

    with open(os.path.join(output_path, 'metrics.json'), "w") as file:
        json.dump(log_dict, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to the full data")
    parser.add_argument("--train-labels-path", type=str, required=True, help="Path to train labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--train-path", type=str, help="Path to training data (if provided, picks training data from this path instead of --data-path")
    parser.add_argument("--model-name-or-path", type=str, default='bert-base-uncased', help="Language Model PROTAUGMENT initializes from")
    parser.add_argument("--lr", default=1e-6, type=float, help="learning rate")
    parser.add_argument("--mixup_lr", default=4e-5, type=float, help="mixup learning rate")
    parser.add_argument("--super-tau", default=1.0, type=float, help="Temperature of the contrastive loss in supervised loss")
    parser.add_argument("--max-len", type=int, default=64, help="maxmium length of text sequence for BERT") 
    parser.add_argument("--mixup_ratio", default=0.9)
    parser.add_argument("--k", default=1)

    # Few-Shot related stuff
    parser.add_argument("--n-support", type=int, default=5, help="Number of support points for each class")
    parser.add_argument("--n-query", type=int, default=1, help="Number of query points for each class")
    parser.add_argument("--n-classes", type=int, default=1, help="Number of classes per episode")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance function to use", choices=("euclidean", "cosine"))

    # Validation & test
    parser.add_argument("--valid-labels-path", type=str, required=True, help="Path to valid labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--test-labels-path", type=str, required=True, help="Path to test labels. This file contains unique names of labels (i.e. one row per label)")
    parser.add_argument("--evaluate-every", type=int, default=100, help="Number of training episodes between each evaluation (on both valid, test)")
    parser.add_argument("--n-test-episodes", type=int, default=600, help="Number of episodes during evaluation (valid, test)")

    # Logging & Saving
    parser.add_argument("--output-path", type=str, default=f'runs/{now()}')
    parser.add_argument("--checkpoint-path", type=str, default=f'checkpoint/{now()}')
    parser.add_argument("--log-every", type=int, default=10, help="Number of training episodes between each logging")
    parser.add_argument("--log-path", type=str, help="Path to the log file.")

    # Training stuff
    parser.add_argument("--max-iter", type=int, default=10000, help="Max number of training episodes")
    parser.add_argument("--early-stop", type=int, default=10, help="Number of worse evaluation steps before stopping. 0=disabled")

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed to set")

    # Supervised loss share
    parser.add_argument("--supervised-loss-share-power", default=1.0, type=float, help="supervised_loss_share = 1 - (x/y) ** <param>")

    args = parser.parse_args()

    logging.basicConfig(filename=args.log_path, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    warnings.simplefilter('ignore')

    handler = logging.FileHandler(args.log_path, mode='w')

    logger.addHandler(handler)

    logger.debug(f"Received args: {json.dumps(args.__dict__, sort_keys=True, ensure_ascii=False, indent=1)}")

    # Set random seed
    set_seeds(args.seed)

    # Check if data path(s) exist
    for arg in [args.data_path, args.train_labels_path, args.valid_labels_path, args.test_labels_path]:
        if arg and not os.path.exists(arg):
            raise FileNotFoundError(f"Data @ {arg} not found.")

    # Run
    run_fsintent(
        data_path=args.data_path,
        train_labels_path=args.train_labels_path,
        train_path=args.train_path,
        model_name_or_path=args.model_name_or_path,
        super_tau=args.super_tau,
        n_support=args.n_support,
        n_query=args.n_query,
        n_classes=args.n_classes,
        lr=args.lr,
        mixup_lr=args.mixup_lr,
        metric=args.metric,
        logger=logger,
        max_len=args.max_len,
        mixup_ratio=args.mixup_ratio,
        k=args.k,

        valid_labels_path=args.valid_labels_path,
        test_labels_path=args.test_labels_path,
        evaluate_every=args.evaluate_every,
        n_test_episodes=args.n_test_episodes,

        output_path=args.output_path,
        checkpoint_path=args.checkpoint_path,
        log_every=args.log_every,
        max_iter=args.max_iter,
        early_stop=args.early_stop,
    )

    # Save config
    with open(os.path.join(args.output_path, "config.json"), "w") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=1)

    

if __name__ == '__main__':
    main()

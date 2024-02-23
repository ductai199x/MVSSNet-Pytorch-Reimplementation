import os
import rich
import argparse
import yaml
import wandb
import pandas as pd

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from typing import *

from data.videofact2_dataset import VideoFact2Dataset
from mvss_plwrapper import MVSS_PLWrapper

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()

ds_root_dir = "/media/nas2/videofact2_data_large"
ds_metadata_path = f"{ds_root_dir}/metadata.csv"


def prepare_model(args: dict[str, Any]) -> MVSS_PLWrapper:
    if args is None:
        raise ValueError("args is None")
    prev_ckpt = args.get("prev_ckpt")
    if prev_ckpt:
        print(f"Loading from checkpoint: {prev_ckpt}...")
        return MVSS_PLWrapper.load_from_checkpoint(prev_ckpt, model_config=args["model_args"], training_config=args["training_args"])
    else:
        return MVSS_PLWrapper(args["model_args"], args["training_args"])


def prepare_logger(args: dict[str, Any]) -> Tuple[Optional[Union[WandbLogger, TensorBoardLogger]], str]:
    if args["fast_dev_run"]:
        return None, None

    if args.get("logger") is None:
        logger_method = "tensorboard"
    else:
        logger_method = args["logger"]
    args_log_dir = args["log_dir"]
    args_version = args["version"]
    args_uid = args["uid"]

    if logger_method == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f"version_{args_version}",
            name=args_log_dir,
            log_graph=True,
        )
        log_path = f"{args_log_dir}/version_{args_version}"
        return logger, log_path
    elif logger_method == "wandb":
        log_path = f"{args_log_dir}/version_{args_version}"
        wandb_path = f"{log_path}/wandb"
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        run_uid = args_uid if args_uid else wandb.util.generate_id()
        logger = WandbLogger(
            project="mvssnet",
            save_dir=log_path,
            version=f"version_{args_version}_{run_uid}",
            name=f"mvssnet_{args_version}_{run_uid}",
            log_model="all",
        )
        return logger, wandb_path
    else:
        raise NotImplementedError(f"Unknown logger method: {logger_method}")


def train(args: argparse.Namespace) -> None:
    # define how the model is loaded in the prepare_model.py file
    model = prepare_model(args.__dict__)
    logger, log_path = prepare_logger(args.__dict__)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_ckpt = ModelCheckpoint(
        dirpath=f"{log_path}/checkpoints",
        monitor="val_det_acc",
        filename=f"{args.pre + '-' if args.pre != '' else ''}{{epoch:02d}}-{{val_det_acc:.4f}}-{{val_loc_f1:.4f}}",
        verbose=True,
        save_last=True,
        save_top_k=2,
        mode="max",
        every_n_epochs=1,
    )
    callbacks = [] if args.fast_dev_run else [ModelSummary(-1), TQDMProgressBar(refresh_rate=1), model_ckpt, lr_monitor]
    if args.fast_dev_run:
        num_gpus = 1
    else:
        num_gpus = "auto" if args.gpus == -1 else args.gpus
    trainer = Trainer(
        accelerator="auto",
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices=num_gpus,
        max_epochs=args.max_epochs,
        logger=logger,
        profiler=None,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        enable_checkpointing=not args.fast_dev_run,
        log_every_n_steps=10,
    )
    if isinstance(logger, WandbLogger):
        logger.watch(model, log="all", log_freq=100)
    trainer.fit(model, ckpt_path=args.prev_ckpt if args.resume else None)


def parse_args(args: argparse.Namespace) -> argparse.Namespace:
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"config file does not exist: {args.config}")
    if args.prev_ckpt and not os.path.exists(args.prev_ckpt):
        raise FileNotFoundError(f"previous checkpoint file does not exist: {args.prev_ckpt}")
    if args.log_dir and not os.path.isdir(args.log_dir):
        print(f"Log dir does not exist: {args.log_dir}. Trying to create it..")
        os.makedirs(args.log_dir)
    if args.resume and not args.prev_ckpt:
        raise ValueError("resume is true but there's no checkpoint specified")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        args.model_args = config["model_args"]
        args.training_args = config["training_args"]
        args.max_epochs = config["training_args"]["max_epochs"]

    rich.print_json(data=args.__dict__)
    return args


def main():
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="the path to a config file",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--prev-ckpt",
        type=str,
        help="the path to a previous checkpoint",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="resume the training progress? False will reset the optimizer state (True/False)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="the version of this model (same as the one saved in log dir)",
        default="0",
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        type=str,
        help="the path to the log directory",
        default="lightning_logs/mvssnet",
    )
    parser.add_argument(
        "-f",
        "--fast-dev-run",
        action="store_true",
        help="fast dev run? (True/Fase)",
    )
    parser.add_argument(
        "--pre",
        type=str,
        help="checkpoint's prefix",
        default="",
    )
    parser.add_argument(
        "--gpus",
        type=lambda x: [int(i) for i in x.split(",")],
        help="specify which GPUs to use (comma-separated list) or leave it alone to use all available GPUs",
        default=-1,
    )
    parser.add_argument(
        "--logger",
        type=str,
        choices=["tensorboard", "wandb"],
        help="logger method (tensorboard/wandb)",
        default="tensorboard",
    )
    parser.add_argument(
        "--uid",
        type=str,
        help="unique id for wandb in case resuming runs with same version",
        default=None,
    )
    args = parser.parse_args()
    args = parse_args(args)

    train(args)


if __name__ == "__main__":
    main()

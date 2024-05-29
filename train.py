from __future__ import annotations

import os
import random
import sys

import dotenv
import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# MixedPrecisionPlugin
from lightning.pytorch.plugins import MixedPrecisionPlugin
from loguru import logger
from omegaconf import DictConfig

from fdlsar import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(
    version_base="1.1",
    config_path="configs/example-configs",
    config_name="train.yaml",
)
def main(config: DictConfig):
    # ------- seeds -------

    # extract and set model and data seeds
    model_seed = config.model_seed if "model_seed" in config else 42
    data_seed = config.data_seed if "data_seed" in config else 42

    logger.info(f"training with model seed {model_seed}")
    logger.info(f"training with data seed {data_seed}")

    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)

    # ------- wandb logging -------

    # set up wandb config
    wandb.config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    # get wandb experiment tags from config
    tags = config.tags if "tags" in config else []
    if isinstance(tags, str):
        tags = tags.split()

    # get experiment name from config
    experiment_name = config.experiment_name
    if "aoi" in config:
        aoi = config.aoi
        # add location information to experiment name
        experiment_name = f"{experiment_name}_{aoi}"
        # add wandb tag for aoi
        tags.append(config.aoi)
    else:
        aoi = "aoi-not-specified"

    # set up wandb logger
    wandb_logger = WandbLogger(
        name=experiment_name,
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode=config.wandb.mode,
        tags=tags,
    )

    # log command to wandb
    log_cmd_wandb = config.log_cmd_wandb if "log_cmd_wandb" in config else False
    logger.info(log_cmd_wandb)
    if log_cmd_wandb:
        cmd = " ".join(sys.argv)
        logger.info(f"Command executed: {cmd}")

    # log config to wandb
    log_config_as = config.log_config_as if "log_config_as" in config else "yaml"
    if log_config_as == "yaml":
        yaml_str = omegaconf.OmegaConf.to_yaml(config)
        logger.debug(f"Hydra-config: {yaml_str}")
    else:
        logger.debug(f"Hydra-config: {config}")

    # ------- dataloader -------

    # Instantiate dataloader
    dataloader = hydra.utils.instantiate(config.dataloader)

    # ------- model -------

    # load checkpoint or instantiate model from scratch
    if "load_checkpoint" in config.keys():
        hr = utils.find_hydra_run_path(
            outputs_dir=config.load_checkpoint.outputs_dir,
            wandb_runid=config.load_checkpoint.wandb_runid,
        )
        logger.info("hydra run path for previous model", hr)
        model = utils.load_ckpt_from_hydra_run(hr)
    else:
        logger.info("instantiating model")
        model = hydra.utils.instantiate(config.model)
        if isinstance(model, tuple):
            logger.info("selecting first model in tuple")
            model = model[0]

    # ------- callbacks -------

    # Checkpoint callback
    # Define the checkpoint callback path
    dirpath = f"checkpoints/"
    # Define the monitored metric
    monitor_metric = (
        config.get("monitor_metric") if config.get("monitor_metric") else "val/loss"
    )
    # Define the checkpoint callback
    val_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor=monitor_metric,
        save_top_k=1,
        mode="min",
        filename=f"{aoi}-{experiment_name}"
        + "-best-monitored-metric-epoch={epoch:02d}",
        auto_insert_metric_name=False,
    )

    callbacks = [
        val_checkpoint_callback,
    ]

    # # ------- training details -------
    #     callbacks = [
    #         val_checkpoint_callback,
    #         linear_probe_callback,
    #         # epoch_checkpoint_callback,
    #     ]
    # else:
    #     callbacks = [
    #         val_checkpoint_callback,
    #         # epoch_checkpoint_callback,
    #     ]

    # ft_scheduler__create_initial_file = (
    #     config.ft_scheduler__create_initial_file
    #     if "ft_scheduler__create_initial_file" in config
    #     else False
    # )
    # if config.get("ft_scheduler"):
    #     if not ft_scheduler__create_initial_file:
    #         logger.info(f"---- using ft_scheduler")
    #         ft_scheduler = FinetuningScheduler(**dict(config.ft_scheduler))
    #         callbacks.append(ft_scheduler)
    #     else:
    #         logger.warning(
    #             f"---- training with ft_scheduler ---****however*** the finetuner is not defined."
    #         )
    #         ft_scheduler = FinetuningScheduler()
    #         callbacks.append(ft_scheduler)
    # else:
    #     logger.info(f"---- not using ft_scheduler")

    # if config.get("log_data_sum"):
    #     callbacks.append(LogDataSum(input=input))

    # if config.get("plot_esawc"):
    #     callbacks.append(LogImagePredictions())

    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0"),
    #     schedule=torch.profiler.schedule(
    #         skip_first=10, wait=5, warmup=1, active=5, repeat=2
    #     ),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # )

    # # train with configured model and dataloader
    # dataloader = hydra.utils.instantiate(config.dataloader)

    # precision = config.get("precision") if config.get("precision") else "32-true"
    # print("ACC", config.get("accumulate_grad_batches"))
    # accumulate_grad_batches = (
    #     config.get("accumulate_grad_batches")
    #     if config.get("accumulate_grad_batches")
    #     else 1
    # )

    # # this is necessary to train using antialiasing torch algorithms
    # # warn_only is used to log the error but not stop the train

    # if config.get("use_deterministic_algorithms"):
    #     torch.use_deterministic_algorithms(True, warn_only=True)

    # # Define the limit of test_batches
    # limit_test_batches = (
    #     1.0 if not hasattr(config, "limit_test_batches") else config.limit_test_batches
    # )

    # devices = config.devices if hasattr(config, "devices") else "auto"
    # check_val_every_n_epoch = (
    #     config.check_val_every_n_epoch
    #     if hasattr(config, "check_val_every_n_epoch")
    #     else 1
    # )

    # if config.get("strategy"):
    #     strategy = config.strategy
    # else:
    #     timeout = datetime.timedelta(minutes=60)
    #     strategy = DDPStrategy(
    #         find_unused_parameters=True,
    #         static_graph=True,
    #         timeout=timeout,
    #     )

    # # strategy = (
    # #     config.get("strategy")
    # #     if config.get("strategy")
    # #     else "ddp_find_unused_parameters_true"
    # # )

    # Define plugins for trainer
    plugins = None

    # Define the precision for the model
    precision = config.get("precision") if config.get("precision") else "32-true"
    if precision == "16-mixed":
        logger.info("---- using mixed precision, adding plugin")
        plugins = [
            MixedPrecisionPlugin(
                "16-mixed", device="cuda", scaler=torch.cuda.amp.GradScaler()
            )
        ]

    # Define whether to accumulate gradients before running optimizer
    accumulate_grad_batches = (
        config.get("accumulate_grad_batches")
        if config.get("accumulate_grad_batches")
        else 1
    )
    # Define whether to use deterministic algorithms
    if config.get("use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Define the limit of test_batches
    limit_test_batches = (
        1.0 if not hasattr(config, "limit_test_batches") else config.limit_test_batches
    )

    # Set device to use
    devices = config.devices if hasattr(config, "devices") else "auto"

    # Define when to run validation loop
    check_val_every_n_epoch = (
        config.check_val_every_n_epoch
        if hasattr(config, "check_val_every_n_epoch")
        else 1
    )
    # Define strategy for trainer
    strategy = (
        config.get("strategy")
        if config.get("strategy")
        else "ddp_find_unused_parameters_true"
    )

    # ------- training -------
    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        # accelerator="gpu",
        strategy=strategy,
        devices=devices,
        plugins=plugins,
        max_epochs=config.max_epochs,
        precision=precision,
        log_every_n_steps=config.log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        fast_dev_run=False,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        limit_test_batches=limit_test_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,  # Disable sanity check validation steps
    )

    trainer.fit(model, dataloader)

    # ------- testing -------
    logger.info(f"---- getting best model from {os.getcwd()}")
    wandb.log({"best_model_dir": os.getcwd()})
    # utils.load_ckpt_from_hydra_run(os.getcwd())

    # trainer.test(model=best_model, datamodule=dataloader)


if __name__ == "__main__":
    main()

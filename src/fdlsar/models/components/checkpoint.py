from __future__ import annotations

import lightning.pytorch as pl
from loguru import logger

from fdlsar import utils


class ModelFromHydraRun(pl.LightningModule):
    def __init__(
        self,
        hydra_run_path,
        model_expression=None,
        freeze=False,
        loading_from_state_dict=True,
        enable_loading_weights: bool = True,
        model_setvars = {}
    ):
        """
        hydra_run_path: see utils.load_ckpt_from_hydra_run
        model_expression: a str that will be evaluated within the model
                          object loaded from the checkpoint

        """
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()

        self.hydra_run_path = hydra_run_path
        self.model_expression = model_expression
        self.freeze = freeze

        # create model
        self.model = utils.load_ckpt_from_hydra_run(
            hydra_run_path,
            loading_from_state_dict=loading_from_state_dict,
            enable_loading_weights=enable_loading_weights,
        )
        if "embedding_x" in dir(self.model):  # for vicreg resnet
            self.output_dim = self.model.embedding_x

        self.model = eval(f"self.model.{model_expression}")

        # freeze
        if freeze:
            try:
                self.model.freeze()
            except:
                logger.info("Model is not PLModule, freezing manually")
                for p in self.model.parameters():
                    p.requires_grad = False

        for k,v in model_setvars.items():
            logger.info(f"---- model setting {k} to {v}")
            exec(f"self.model.{k} = {v}")


        logger.info("-------------------------")
        logger.info(f"freezing {freeze}")
        logger.info("-------------------------")

        if "output_dim" in dir(self.model):
            self.output_dim = self.model.output_dim

    def forward(self, x):
        return self.model(x)
# Standard library imports
from foundational_ssm.utils import h5_to_dict
import os
import multiprocessing as mp
import logging
import hydra
from dotenv import load_dotenv

# Third-party imports
import numpy as np
import wandb
from jax import random as jr
import equinox as eqx

# Foundational SSM imports
from omegaconf import OmegaConf

from foundational_ssm.loaders import get_brainset_train_val_loaders

from foundational_ssm.utils.downstream_utils import (
    mse_loss_downstream,
    train_one_epoch_brainsets,
    validate_one_epoch_brainsets,
    load_training_state,
)

from foundational_ssm.utils.wandb_utils_jax import (
    save_checkpoint_wandb,
    add_alias_to_checkpoint,
    count_parameters,
)
import multiprocessing as mp

load_dotenv()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="pm_transfer", version_base="1.3")
def main(cfg: OmegaConf):
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(filename="downstream_decoding_rtt.log", level=logging.INFO)

    best_r2_score = 0

    key, train_key, val_key = jr.split(jr.PRNGKey(cfg.rng_seed), 3)
    prepend_history = cfg.prepend_history
    skip_timesteps = int(prepend_history * cfg.sampling_rate)
    _, train_loader, _, val_loader, max_neural_units = get_brainset_train_val_loaders(
        cfg.dataset_args, cfg.train_loader, cfg.val_loader, prepend_history
    )
    cfg, model, state, opt, opt_state, lr_scheduler = load_training_state(
        cfg, max_neural_units
    )

    model_num_params = count_parameters(model)
    wandb.log({"model/num_params": model_num_params}, step=0)

    current_step = 0
    epoch_loss = 1e10
    for epoch in range(0, cfg.training.epochs):
        if cfg.training.save_checkpoints and epoch % cfg.training.checkpoint_every == 0:
            metadata = {"train_loss": epoch_loss, "epoch": epoch, "step": current_step}
            checkpoint_artifact = save_checkpoint_wandb(
                model, state, opt_state, metadata
            )

        if epoch % cfg.training.log_val_every == 0:
            if "checkpoint_artifact" in locals():
                add_alias_to_checkpoint(checkpoint_artifact, f"epoch_{epoch}")

            logger.info(f"Running validation for epoch {epoch}")
            metrics = validate_one_epoch_brainsets(
                val_loader, model, state, skip_timesteps
            )
            metrics["epoch"] = epoch
            wandb.log(metrics, step=current_step)
            current_r2 = metrics.get(f"val/r2", 0.0)

            if current_r2 > best_r2_score:
                best_r2_score = current_r2
                logger.info(f"New best R² score: {best_r2_score:.4f} at epoch {epoch}")
                if "checkpoint_artifact" in locals():
                    add_alias_to_checkpoint(checkpoint_artifact, "best", metrics)

        train_key, subkey = jr.split(train_key)
        model, state, opt_state, current_step, epoch_loss = train_one_epoch_brainsets(
            train_loader,
            model,
            state,
            mse_loss_downstream,
            opt,
            opt_state,
            subkey,
            lr_scheduler,
            current_step,
            epoch,
            skip_timesteps,
        )

    wandb.log(
        {f"final/r2/{cfg.dataset_args.recording_id.split('/')[-1]}/mean": best_r2_score}
    )
    wandb.finish()


if __name__ == "__main__":
    main()

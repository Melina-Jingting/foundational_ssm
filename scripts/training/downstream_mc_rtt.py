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

from foundational_ssm.utils.downstream_utils import (
    mse_loss_downstream,
    train_one_epoch,
    validate_one_epoch,
    log_predictions_and_activations,
    get_rtt_datasets,
    create_model_and_state,
    load_training_state
)
from foundational_ssm.utils.training_utils import (
    create_optimizer_and_state
)
from foundational_ssm.utils.wandb_utils_jax import (
    save_checkpoint_wandb,
    add_alias_to_checkpoint,
    count_parameters
)
from foundational_ssm.models import SSMDownstreamDecoder
import multiprocessing as mp

load_dotenv()
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="rtt_transfer", version_base="1.3")
def main(cfg: OmegaConf):
    
    
    
    mp.set_start_method("spawn", force=True)
    logging.basicConfig(filename='downstream_decoding_rtt.log', level=logging.INFO)
    
    best_r2_score = 0
    key, train_key, val_key = jr.split(jr.PRNGKey(cfg.rng_seed), 3)
    train_data, val_data, data = get_rtt_datasets(cfg.dataset, val_key)
    # model, state, model_cfg = create_model_and_state(cfg.model)
    # opt, opt_state, lr_scheduler = create_optimizer_and_state(model, 
    #                                                         cfg.optimizer,#   getattr(cfg, 'optimizer', OmegaConf.load(cfg.model.cfg).optimizer),
    #                                                           model_cfg)
    
    # cfg.model = model_cfg # update metadata for wandb run
    # model_num_params = count_parameters(model)
    # wandb.init(
    #     project=cfg.wandb.project,
    #     entity=cfg.wandb.entity,
    #     tags=cfg.wandb.tags,
    #     config=OmegaConf.to_container(cfg),
    #     name=f"l{cfg.model.ssm_num_layers}_scratch_{cfg.optimizer.mode}"
    # )
    # wandb.log({"model/num_params": model_num_params}, step=0)
    
    cfg, model, state, opt, opt_state, lr_scheduler = load_training_state(cfg, 130)
    model_num_params = count_parameters(model)
    wandb.log({"model/num_params": model_num_params}, step=0)
    
    current_step = 0
    epoch_loss = 1e10
    for epoch in range(0, cfg.training.epochs):
        if cfg.training.save_checkpoints and epoch % cfg.training.checkpoint_every == 0:
            metadata = {
                'train_loss': epoch_loss,
                'epoch': epoch,
                'step': current_step
            }
            checkpoint_artifact = save_checkpoint_wandb(model, state, opt_state, metadata)

        if epoch % cfg.training.log_val_every == 0:
            if 'checkpoint_artifact' in locals():
                add_alias_to_checkpoint(checkpoint_artifact, f'epoch_{epoch}')

            logger.info(f"Running validation for epoch {epoch}")
            metrics = validate_one_epoch(val_data, model, state, epoch, current_step, cfg.dataset.skip_timesteps)
            current_r2 = metrics.get(f'val/r2', 0.0)
            
            if current_r2 > best_r2_score:
                best_r2_score = current_r2
                logger.info(f"New best R² score: {best_r2_score:.4f} at epoch {epoch}")
                if 'checkpoint_artifact' in locals():
                    add_alias_to_checkpoint(checkpoint_artifact, 'best', metrics)
                    
        if cfg.training.save_activations and epoch % cfg.training.log_pred_and_activations_every == 0:
            log_predictions_and_activations(model, state, data, cfg, epoch, current_step, wandb.run.id, 'rtt_prepend_279ms')
        
        train_key, subkey = jr.split(train_key)
        model, state, opt_state, current_step, epoch_loss = train_one_epoch(
            train_data, model, state, mse_loss_downstream, opt, opt_state, lr_scheduler, current_step, cfg.dataset.skip_timesteps, cfg.dataset.batch_size, subkey
        )

    wandb.log({"final/best_r2": best_r2_score})
    wandb.finish()


if __name__ == "__main__":
    main()
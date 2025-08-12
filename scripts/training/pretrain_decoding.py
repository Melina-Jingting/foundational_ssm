import logging
import wandb
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
import multiprocessing as mp
import jax.random as jr

# Foundational SSM core imports
from foundational_ssm.models import SSMFoundationalDecoder
from foundational_ssm.loaders import get_brainset_train_val_loaders
from foundational_ssm.utils.pretrain_utils import (
    train_one_epoch,
    validate_one_epoch,
    load_training_state,
    mse_loss_foundational,
)
from foundational_ssm.utils.wandb_utils_jax import (
    save_checkpoint_wandb,
    add_alias_to_checkpoint,
)

load_dotenv()
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    
    loss_fn = mse_loss_foundational
    
    cfg, model, state, opt, opt_state, start_epoch, lr_scheduler, current_step, best_r2_score \
        = load_training_state(cfg, model_cls=SSMFoundationalDecoder, wandb_resume_run_id=cfg.wandb.resume_run_id)
        
    _, train_loader, _, val_loader = get_brainset_train_val_loaders(
            cfg.train_loader,
            cfg.val_loader,
            cfg.dataset_cfg
        )
    
    key = jr.PRNGKey(cfg.rng_seed)
    epoch_loss = 1e10
    for epoch in range(start_epoch, cfg.training.epochs):

        if epoch % cfg.training.checkpoint_every == 0:
            checkpoint_metadata = {
                'train_loss': epoch_loss,
                'epoch': epoch,
                'step': current_step
            }
            checkpoint_artifact = save_checkpoint_wandb(model, state, opt_state, checkpoint_metadata)
    
        if epoch % cfg.training.log_val_every == 0:
            add_alias_to_checkpoint(checkpoint_artifact, f'epoch_{epoch}')
            logger.info(f"Running validation for epoch {epoch}")
            metrics = validate_one_epoch(val_loader, model, state, epoch, current_step, cfg.skip_timesteps)
            current_r2_avg = metrics.get('val/r2_avg', 0.0)
            if current_r2_avg > best_r2_score:
                best_r2_score = current_r2_avg
                logger.info(f"New best R² score: {best_r2_score:.4f} at epoch {epoch}")
                add_alias_to_checkpoint(checkpoint_artifact,  'best', metadata = metrics)
        
        # =================================================================
        # Training step
        # =================================================================
        key, train_key = jr.split(key)
        logger.info(f"Running training for epoch {epoch}") 
        model, state, opt_state, current_step, epoch_loss = train_one_epoch(
                train_loader, model, state, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step, epoch, cfg.skip_timesteps
            )
        
    wandb.log({
        "final/best_r2_avg": best_r2_score
    }, step=current_step)
    wandb.finish()
    
    logger.info("Training completed successfully")
            
if __name__ == "__main__":
    main()
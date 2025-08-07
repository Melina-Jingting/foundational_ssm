# Standard library imports
import os
import json
import tempfile
import multiprocessing as mp
import logging
import time
from functools import partial

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import jax
import jax.numpy as jnp
from jax import random as jr
from torch.utils.data import DataLoader
import optax
import equinox as eqx

# Foundational SSM imports
from omegaconf import OmegaConf
from foundational_ssm.constants import (
    DATA_ROOT,
    parse_session_id,
    DATASET_IDX_TO_GROUP_SHORT,
)
from foundational_ssm.loaders import get_brainset_train_val_loaders
from foundational_ssm.utils import load_model_and_state_wandb, save_checkpoint_wandb, transfer_foundational_to_downstream
from foundational_ssm.metrics import compute_r2_standard
from foundational_ssm.utils.training import (
    make_step_downstream,
    mse_loss_downstream,
    get_filter_spec,
)
from foundational_ssm.utils.training_utils import (
    log_batch_metrics, track_batch_timing, setup_wandb_metrics
)
from foundational_ssm.samplers import SequentialFixedWindowSampler
from foundational_ssm.collate import pad_collate
from foundational_ssm.models import SSMFoundationalDecoder, SSMDownstreamDecoder

def train_one_batch(batch, model, state, filter_spec, loss_fn, opt, opt_state, train_key, lr_scheduler, current_step):
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    mask = batch["mask"]
    
    key, subkey = jr.split(train_key)
    model, state, opt_state, loss_value, grads = make_step_downstream(model, state, inputs, targets, mask, key, filter_spec, loss_fn, opt, opt_state)
    current_lr = lr_scheduler(current_step)
    wandb.log({
        "train/loss": loss_value,
        "train/learning_rate": current_lr,
    }, step=current_step)
    
    return model, state, opt_state, loss_value

def train_one_epoch(train_loader, model, state, rng_key, filter_spec, loss_fn, opt, opt_state, lr_scheduler, current_step, epoch):    
    epoch_loss = 0
    batch_count = 0
    minute_start_time = time.time()
    prev_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        data_load_time = time.time() - prev_time
        batch_process_start = time.time()
        
        model, state, opt_state, loss_value = train_one_batch(
            batch, model, state, filter_spec, loss_fn, opt, opt_state, rng_key, lr_scheduler, current_step
        )
        batch_process_end = time.time()
        batch_process_time = batch_process_end - batch_process_start
        
        log_batch_metrics(data_load_time, batch_process_time, epoch, current_step)
        epoch_loss += loss_value
        batch_count += 1
        current_time = time.time()
        batch_count, minute_start_time = track_batch_timing(batch_count, minute_start_time, current_time, current_step)
        prev_time = time.time()
        current_step += 1
    
    wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch}, step=current_step)
    return model, state, opt_state, current_step, epoch_loss

def validate_one_epoch(val_loader, model, state, epoch, current_step, dataset_name):
    logger.info("Validating one epoch")
    metrics = {}  # New: store metrics per group
    all_preds = []
    all_targets = []
    val_start_time = time.time()
    for batch_idx, batch in enumerate(val_loader):
        batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}
        inputs = batch["neural_input"]
        targets = batch["behavior_input"]
        mask = batch["mask"]
        mask = mask[..., None]
        preds, state = jax.vmap(model, axis_name="batch", in_axes=(0, None, None, None), out_axes=(0, None))(inputs, state, jr.PRNGKey(0), True)
        all_preds.append(jnp.where(mask, preds, 0))
        all_targets.append(jnp.where(mask, targets, 0))
        prev_time = time.time()

    all_preds = jnp.concatenate(all_preds, axis=0)
    all_targets = jnp.concatenate(all_targets, axis=0)
    
    r2_score = compute_r2_standard(all_preds, all_targets)
    metrics[f'val/r2_{dataset_name}'] = float(r2_score)

    # Log validation timing and resources
    val_end_time = time.time()
    val_time = val_end_time - val_start_time
    metrics[f'val/time_{dataset_name}'] = val_time
    metrics['epoch'] = epoch

    wandb.log(metrics, step=current_step)
    return metrics

def add_best_alias_to_checkpoint(checkpoint_artifact, metadata):
    checkpoint_artifact.wait()
    if 'best' not in checkpoint_artifact.aliases:
        checkpoint_artifact.aliases.append('best')
    checkpoint_artifact.metadata.update(metadata)
    checkpoint_artifact.save()
    
    
config_path = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/configs/downstream.yaml"
cfg = OmegaConf.load(config_path) 
api = wandb.Api()
logging.basicConfig(filename='pretrain_decoding.log', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
tempdir = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/wandb_artifacts"
best_r2_score = 0

key, train_key, val_key = jr.split(jr.PRNGKey(cfg.rng_seed), 3)

for dataset_name, config in cfg.downstream_datasets.items():
    for model_name, model_cfg in cfg.models.items():
        for downstream_mode_name, downstream_mode_cfg  in cfg.downstream_modes.items():
            if hasattr(model_cfg, 'checkpoint'):
                artifact_full_name = model_cfg.checkpoint
                # ===========================================================
                # load_checkpoint 
                # ===========================================================
                artifact = api.artifact(artifact_full_name, type="checkpoint")
                foundational_run = artifact.logged_by()
                foundational_run_cfg = OmegaConf.create(foundational_run.config)
                
                foundational_model = SSMFoundationalDecoder(
                        **foundational_run_cfg.model
                    )
                
                downstream_model_cfg = foundational_run_cfg.model.copy()
                downstream_model_cfg.update({'input_dim':130})
                downstream_model = SSMDownstreamDecoder(**downstream_model_cfg)

                # ===================================================================================
                #  Load foundational model from checkpoint and transfer SSM layers to downstream
                # ===================================================================================
                if downstream_mode_cfg.from_scratch == False:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        artifact.download(temp_dir)
                        
                        # Find the checkpoint file in the downloaded directory
                        checkpoint_files = [f for f in os.listdir(temp_dir) if f.endswith('.ckpt')]
                        if not checkpoint_files:
                            print(f"Available files in {temp_dir}: {os.listdir(temp_dir)}")
                            raise FileNotFoundError(f"No checkpoint file found in {temp_dir}. Available files: {os.listdir(temp_dir)}")
                        
                        checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
                        print(f"Loading checkpoint from: {checkpoint_path}")
                        
                        with open(checkpoint_path, 'rb') as f:
                            meta = json.loads(f.readline().decode())
                            foundational_model = eqx.tree_deserialise_leaves(f, foundational_model)
                        
                    downstream_model = transfer_foundational_to_downstream(foundational_model, downstream_model)
            
            if hasattr(model_cfg, 'cfg'):
                # ===========================================================
                # load_model_cfg 
                # ===========================================================
                downstream_model_cfg = model_cfg.cfg
                downstream_model = SSMDownstreamDecoder(**downstream_model_cfg)
                if downstream_mode_cfg.from_scratch == False:
                    print("Not training from scratch only available if model config has checkpoint")
                    continue
                
            downstream_state = eqx.nn.State(downstream_model)
            
            run_name = f"{cfg.wandb.run_prefix}_{dataset_name}_{downstream_mode_name}_{model_name}"
            cfg.update({'downstream_model_cfg': downstream_model_cfg})
            wandb.init(project=cfg.wandb.project, name=run_name, config=dict(cfg)) 
        
            filter_spec = get_filter_spec(
                downstream_model,
                **cfg.filter_spec
            )
            
            lr_scheduler = lambda step: cfg.optimizer.lr
            
            opt = optax.chain(
                optax.adamw(learning_rate=lr_scheduler, weight_decay=cfg.optimizer.weight_decay)
            )
            opt_state = opt.init(eqx.filter(downstream_model, filter_spec))
            
            current_step = 0
            
            # ===========================================================
            # load datasets
            # ===========================================================
            cfg.train_loader.dataset_args.update({'config':config})
            cfg.val_loader.dataset_args.update({'config':config})
            train_dataset, train_loader, val_dataset, val_loader = get_brainset_train_val_loaders(
                cfg.train_loader,
                cfg.val_loader,
                data_root = '../' + DATA_ROOT
            )
            
            for epoch in range(1, cfg.training.epochs + 1):
                train_key, subkey = jr.split(train_key)
                logger.info(f"Running training for epoch {epoch}")
                downstream_model, downstream_state, opt_state, current_step, epoch_loss = train_one_epoch(train_loader, downstream_model, downstream_state, subkey, filter_spec, mse_loss_downstream, opt, opt_state, lr_scheduler, current_step, epoch)

                if epoch % cfg.training.checkpoint_every == 0:
                    metadata = {
                        'train_loss': epoch_loss
                    }
                    logger.info(f"Saving checkpoint for epoch {epoch}")
                    checkpoint_artifact = save_checkpoint_wandb(downstream_model, downstream_state, opt_state, epoch, current_step, metadata, run_name)

                if epoch % cfg.training.log_val_every == 0:
                    logger.info(f"Running validation for epoch {epoch}")
                    metrics = validate_one_epoch(val_loader, downstream_model, downstream_state, epoch, current_step, dataset_name)
                    
                    # Track best R² score
                    current_r2_avg = metrics.get(f'val/r2_{dataset_name}', 0.0)
                    if current_r2_avg > best_r2_score:
                        best_r2_score = current_r2_avg
                        logger.info(f"New best R² score: {best_r2_score:.4f} at epoch {epoch}")
                        add_best_alias_to_checkpoint(checkpoint_artifact, metrics)
            

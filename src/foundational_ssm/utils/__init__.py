"""
Utility functions for the foundational_ssm package.
"""

# Import and expose key functions from misc.py
from .misc import (
    generate_sinusoidal_position_embs,
    load_pretrained,
    reinit_vocab,
    move_to_gpu,
    get_dataset_config,
)

# Import and expose key functions from wandb_utils.py
from .wandb_utils_jax import (
    log_model_params_and_grads_wandb,
    save_model_wandb,
    load_model_wandb,
)

# Import and expose profiling utilities
from .profiling import (
    TrainingProfiler,
    get_profiler,
    profile_operation,
    log_profiling_metrics,
    profile_jax_loss_fn,
    profile_make_step,
    profile_data_loader,
    profile_training_loop,
)

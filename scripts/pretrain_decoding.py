import os
import sys
import warnings
import logging
import torch

# Suppress warnings and logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

# Core imports
from foundational_ssm.utils import get_train_val_loaders, get_dataset_config
from foundational_ssm.data_preprocessing import bin_spikes, map_binned_features_to_global, smooth_spikes
# from foundational_ssm.models import SSMFoundational
from foundational_ssm.loss import CombinedLoss
from foundational_ssm.metrics import ValidationMetrics

from omegaconf import OmegaConf

from temporaldata import Data
from typing import List, Dict

from foundational_ssm.utils.transform import transform_neural_behavior_sample



config_path = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/configs/cmt.yaml"
config = OmegaConf.load(config_path) 

# Load dataset
train_dataset, train_loader, val_dataset, val_loader = get_train_val_loaders(
    train_config=get_dataset_config(
        config.dataset.name,
        subjects=config.dataset.subjects
    ),
    batch_size=config.dataset.batch_size
)
transform = transform_neural_behavior_sample
train_dataset.transform = transform
val_dataset.transform = transform



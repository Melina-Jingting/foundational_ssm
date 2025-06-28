
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import jax
from jax.tree_util import tree_map 

from foundational_ssm.constants import DATASET_GROUP_DIMS, parse_session_id, DATA_ROOT, DATASET_GROUP_TO_IDX
from foundational_ssm.data_utils.spikes import bin_spikes, smooth_spikes
from torch_brain.data import Dataset, collate, chain
from .samplers import GroupedRandomFixedWindowSampler, GroupedSequentialFixedWindowSampler
from torch.utils.data.dataloader import default_collate


def get_dataset_config(
    brainset, 
    sessions=None,
    subjects=None,
    exclude_subjects=None, 
    exclude_sessions=None
):
    brainset_norms = {
        "perich_miller_population_2018": {
            "mean": 0.0,
            "std": 20.0
        }
    }

    config = f"""
    - selection:
      - brainset: {brainset}"""

    # Add sessions if provided
    if sessions is not None:
        if not isinstance(sessions, list):
            sessions = [sessions]
        config += "\n        sessions:"
        for session in sessions:
            config += f"\n          - {session}"

    # Add subjects if provided
    if subjects is not None:
        config += "\n        subjects:"
        for subj in subjects:
            config += f"\n          - {subj}"

    # Add exclude clauses if provided
    if exclude_subjects is not None or exclude_sessions is not None:
        config += "\n        exclude:"
        if exclude_subjects is not None:
            if not isinstance(exclude_subjects, list):
                exclude_subjects = [exclude_subjects]
            config += "\n          subjects:"
            for subj in exclude_subjects:
                config += f"\n            - {subj}"
        if exclude_sessions is not None:
            if not isinstance(exclude_sessions, list):
                exclude_sessions = [exclude_sessions]
            config += "\n          sessions:"
            for sess in exclude_sessions:
                config += f"\n            - {sess}"

    config += f"""
      config:
        readout:
          readout_id: cursor_velocity_2d
          normalize_mean: {brainset_norms[brainset]["mean"]}
          normalize_std: {brainset_norms[brainset]["std"]}
          metrics:
            - metric:
                _target_: torchmetrics.R2Score
    """

    config = OmegaConf.create(config)

    return config

def _ensure_dim(arr: np.ndarray, target_dim: int, *, axis: int = 1) -> np.ndarray:
    """Crop or zero-pad *arr* along *axis* to match *target_dim*.

    This is a thin wrapper around :pymod:`numpy` slicing and :func:`numpy.pad` that
    avoids several conditional blocks in the main routine.
    """
    current_dim = arr.shape[axis]
    if current_dim == target_dim:
        return arr  # nothing to do
    if current_dim > target_dim:
        # Crop
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(None, target_dim)
        return arr[tuple(slicer)]
    # Pad (current_dim < target_dim)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_dim - current_dim)
    return np.pad(arr, pad_width, mode="constant")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def transform_brainsets_to_fixed_dim_samples(
    data: Any,
    *,
    sampling_rate: int = 100,
    num_timesteps: int = 100,
    kern_sd_ms: int = 40,
    bin_width: int = 5,
) -> Dict[str, torch.Tensor | str]:
    """Convert a *temporaldata* sample to a dictionary of Torch tensors.

    The function takes care of binning & smoothing spikes, cropping/padding neural
    and behavioural features to a globally consistent dimensionality that depends
    on the *(dataset, subject, task)* triple.

    Parameters
    ----------
    data: temporaldata.Data
        Sample returned by **torch-brain**/**temporaldata**.
    sampling_rate: int, default=100
        Target sampling rate *Hz* used for binning.
    num_timesteps: int, default=100
        Length of the temporal window after binning.
    kern_sd_ms: int, default=40
        Standard deviation of the Gaussian kernel (in ms) for smoothing spikes.
    bin_width: int, default=5
        Width of the time bins (in ms) used by :func:`smooth_spikes`.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with keys ``neural_input``, ``behavior_input``, ``session_id``
        and ``subject_id``.
    """

    # ------------------------------------------------------------------
    # 1. Bin + smooth spikes
    # ------------------------------------------------------------------
    unit_ids = data.units.id
    binned_spikes = bin_spikes(
        spikes=data.spikes,
        num_units=len(unit_ids),
        bin_size=1 / sampling_rate,
        num_bins=num_timesteps,
    )  # shape: (units, timesteps)

    # Transpose so that time is first → (timesteps, units)
    binned_spikes = binned_spikes.T

    smoothed_spikes = smooth_spikes(
        binned_spikes,
        kern_sd_ms=kern_sd_ms,
        bin_width=bin_width,
    )

    # ------------------------------------------------------------------
    # 2. Prepare behaviour signal (cursor velocity)
    # ------------------------------------------------------------------
    behavior_input = data.cursor.vel  # np.ndarray, (timesteps?, features)

    # Crop/pad time dimension *first* so that we can treat both signals equally
    if behavior_input.shape[0] > num_timesteps:
        behavior_input = behavior_input[:num_timesteps]
    elif behavior_input.shape[0] < num_timesteps:
        behavior_input = np.pad(
            behavior_input,
            ((0, num_timesteps - behavior_input.shape[0]), (0, 0)),
            mode="constant",
        )

    # ------------------------------------------------------------------
    # 3. Align channel dimensions based on (dataset, subject, task)
    # ------------------------------------------------------------------
    dataset, subject, task = parse_session_id(data.session.id)
    group_tuple = (dataset, subject, task)
    group_idx = DATASET_GROUP_TO_IDX[group_tuple]

    max_raw_input_dim = max(dims[0] for dims in DATASET_GROUP_DIMS.values())
    max_raw_output_dim = max(dims[1] for dims in DATASET_GROUP_DIMS.values())
    smoothed_spikes = _ensure_dim(smoothed_spikes, max_raw_input_dim, axis=1)
    behavior_input = _ensure_dim(behavior_input, max_raw_output_dim, axis=1)

    # ------------------------------------------------------------------
    # 4. Pack into torch tensors
    # ------------------------------------------------------------------
    return {
        "neural_input": torch.as_tensor(smoothed_spikes, dtype=torch.float32),
        "behavior_input": torch.as_tensor(behavior_input, dtype=torch.float32),
        "dataset_group_idx": torch.as_tensor(group_idx, dtype=torch.int32)
    } 
    
def jax_collate_fn(batch):
    """
    Collate function that converts all torch.Tensors in a batch (dict or list of dicts)
    to numpy arrays, recursively.
    """
    collated = default_collate(batch)
    return tree_map(
        lambda x: x.numpy() if isinstance(x, torch.Tensor) else x,
        collated
    )

def get_train_val_loaders(recording_id=None, train_config=None, val_config=None, batch_size=32, seed=0, root=DATA_ROOT, transform_fn=transform_brainsets_to_fixed_dim_samples, collate_fn=jax_collate_fn):
    """Sets up train and validation Datasets, Samplers, and DataLoaders
    """
    # -- Train --
    train_dataset = Dataset(
        root=root,                # root directory where .h5 files are found
        recording_id=recording_id,  # you either specify a single recording ID
        config=train_config,                 # or a config for multi-session training / more complex configs
        split="train",
    )
    # We use a random sampler to improve generalization during training
    train_sampling_intervals = train_dataset.get_sampling_intervals()
    train_sampler = GroupedRandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=1.0,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(42)
    )
    # Finally combine them in a dataloader
    train_loader = DataLoader(
        dataset=train_dataset,      # dataset
        batch_sampler=train_sampler,      # sampler
        collate_fn=collate_fn,         # the collator
        num_workers=4,              # data sample processing (slicing, transforms, tokenization) happens in parallel; this sets the amount of that parallelization
        pin_memory=True,
    )

    # -- Validation --
    if val_config is None:
        val_config = train_config  # if no validation config is provided, use the training config
    
    val_dataset = Dataset(
        root=root,
        recording_id=recording_id,
        config=val_config,
        split="valid",
    )
    # For validation we don't randomize samples for reproducibility
    val_sampling_intervals = val_dataset.get_sampling_intervals()
    val_sampler = GroupedSequentialFixedWindowSampler(
        sampling_intervals=val_sampling_intervals,
        window_length=1.0,
        batch_size=batch_size,
        generator=torch.Generator().manual_seed(42)
    )
    # Combine them in a dataloader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    train_dataset.disable_data_leakage_check()
    val_dataset.disable_data_leakage_check()
    train_dataset.transform = transform_fn
    val_dataset.transform = transform_fn

    return train_dataset, train_loader, val_dataset, val_loader
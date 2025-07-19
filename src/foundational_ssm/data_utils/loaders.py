from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import jax
from jax.tree_util import tree_map 
from functools import partial

from foundational_ssm.constants import (
    DATASET_GROUP_TO_IDX,
    MAX_NEURAL_UNITS,
    MAX_BEHAVIOR_DIM,
    DATA_ROOT,
    MC_MAZE_CONFIG,
    MC_RTT_CONFIG,
    NLB_CONFIGS,
    NLB_DATA_ROOT,
    parse_session_id,
)
from foundational_ssm.data_utils.spikes import bin_spikes, smooth_spikes
from foundational_ssm.data_utils.dataset import TorchBrainDataset
from foundational_ssm.data_utils.samplers import RandomFixedWindowSampler, SequentialFixedWindowSampler, TrialSampler
import h5py
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import Dataset

def h5_to_dict(h5obj):
    """Recursive function that reads HDF5 file to dict

    Parameters
    ----------
    h5obj : h5py.File or h5py.Group
        File or group object to load into a dict
    
    Returns
    -------
    dict of np.array
        Dict mapping h5obj keys to arrays
        or other dicts
    """
    data_dict = {}
    for key in h5obj.keys():
        if isinstance(h5obj[key], h5py.Group):
            data_dict[key] = h5_to_dict(h5obj[key])
        else:
            data_dict[key] = h5obj[key][()]
    return data_dict

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

def pad_collate(batch, fixed_seq_len=None):
    # Assume batch is a list of dicts with keys: 'neural_input', 'behavior_input', etc.
    # Each 'neural_input' is a tensor of shape (timesteps, units)
    neural_inputs = [item['neural_input'].squeeze(0) for item in batch]  # (timesteps, units)
    behavioral_inputs = [item['behavior_input'].squeeze(0) for item in batch]
    
    # Determine the fixed sequence length
    if fixed_seq_len is None:
        max_len = max(x.shape[0] for x in neural_inputs)
    else:
        max_len = fixed_seq_len

    # Pad or truncate each sequence to fixed length
    def pad_or_truncate(tensor, max_len):
        seq_len = tensor.shape[0]
        if seq_len == max_len:
            return tensor
        elif seq_len > max_len:
            return tensor[:max_len]
        else:
            pad_shape = (max_len - seq_len,) + tensor.shape[1:]
            pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=0)

    padded_neural = torch.stack([pad_or_truncate(x, max_len) for x in neural_inputs if x is not None])  # (batch, max_len, units)
    padded_behavior = torch.stack([pad_or_truncate(x, max_len) for x in behavioral_inputs if x is not None])

    # Create mask: 1 for real data, 0 for padding
    lengths = [x.shape[0] for x in neural_inputs]
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :min(l, max_len)] = 1

    # Stack other fields (e.g., dataset_group_idx)
    dataset_group_idx = torch.stack([item['dataset_group_idx'] for item in batch])
    
    return {
        'neural_input': padded_neural,
        'behavior_input': padded_behavior,
        'mask': mask,
        'dataset_group_idx': dataset_group_idx,
        # add other fields as needed
    }

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def transform_brainsets_regular_time_series_smoothed(
    data: Any,
    *,
    sampling_rate: int = 200,
    kern_sd_ms: int = 20,
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
    sampling_window_ms: int, default=1000   
        Length of the temporal window after binning.
    kern_sd_ms: int, default=20
        Standard deviation of the Gaussian kernel (in ms) for smoothing spikes.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with keys ``neural_input``, ``behavior_input``, ``session_id``
        and ``subject_id``.
    """

    bin_size_ms = int(1000 / sampling_rate)
    

    # ------------------------------------------------------------------
    # 1. Get smoothed spikes
    # ------------------------------------------------------------------
    if hasattr(data, "smoothed_binned_spikes"):
        smoothed_spikes = data.smoothed_binned_spikes.data
    else:
        binned_spikes, _ = data.spikes.get_regular_time_series_array(
            sampling_rate=sampling_rate,
            raw_array_name="unit_index",
            is_index=True,
        )  # shape: (timesteps, units)

        smoothed_spikes = smooth_spikes(
            binned_spikes,
            kern_sd_ms=kern_sd_ms,
            bin_size_ms=bin_size_ms,
            time_axis=0,
        )

    # ------------------------------------------------------------------
    # 2. Prepare behaviour signal (cursor velocity)
    # ------------------------------------------------------------------
    if hasattr(data, "vel_regular"):
        behavior_input = data.vel_regular.data
    else:
        if data.session.id.startswith("pei_pandarinath_nlb_2021"):
            behavior_input, _ = data.hand.get_regular_time_series_array(
                sampling_rate=sampling_rate,
                raw_array_name="vel",
            )  # np.ndarray, (timesteps, features)
        else:
            behavior_input, _ = data.cursor.get_regular_time_series_array(
                sampling_rate=sampling_rate,
                raw_array_name="vel",
            )  # np.ndarray, (timesteps, features)

    # ------------------------------------------------------------------
    # 3. Drop timesteps where behavior has inf/NaN values
    # ------------------------------------------------------------------
    # Check for inf/NaN in behavior_input
    valid_mask = ~(np.isinf(behavior_input).any(axis=1) | np.isnan(behavior_input).any(axis=1))
    if not valid_mask.all():
        behavior_input = behavior_input[valid_mask]
        smoothed_spikes = smoothed_spikes[valid_mask]
    
    # ------------------------------------------------------------------
    # 4. Align channel dimensions based on (dataset, subject, task)
    # ------------------------------------------------------------------
    dataset, subject, task = parse_session_id(data.session.id)
    group_tuple = (dataset, subject, task)
    group_idx = DATASET_GROUP_TO_IDX[group_tuple]
    smoothed_spikes = _ensure_dim(smoothed_spikes, MAX_NEURAL_UNITS, axis=1)
    behavior_input = _ensure_dim(behavior_input, MAX_BEHAVIOR_DIM, axis=1)
    
    # ------------------------------------------------------------------
    # 5. Pack into torch tensors
    # ------------------------------------------------------------------
    return {
        "neural_input": torch.as_tensor(smoothed_spikes, dtype=torch.float32),
        "behavior_input": torch.as_tensor(behavior_input, dtype=torch.float32),
        "dataset_group_idx": torch.as_tensor(group_idx, dtype=torch.int32),
    }
    
# def transform_brainsets_to_fixed_dim_samples(
#     data: Any,
#     sampling_rate: int = 100,
#     sampling_window_ms: int = 1000
# ) -> Dict[str, torch.Tensor | str]:
#     """Convert a *temporaldata* sample to a dictionary of Torch tensors.

#     The function takes care of binning & smoothing spikes, cropping/padding neural
#     and behavioural features to a globally consistent dimensionality that depends
#     on the *(dataset, subject, task)* triple.

#     Parameters
#     ----------
#     data: temporaldata.Data
#         Sample returned by **torch-brain**/**temporaldata**.
#     sampling_rate: int, default=100
#         Target sampling rate *Hz* used for binning.
#     sampling_window_ms: int, default=1000   
#         Length of the temporal window after binning.
#     kern_sd_ms: int, default=20
#         Standard deviation of the Gaussian kernel (in ms) for smoothing spikes.

#     Returns
#     -------
#     Dict[str, torch.Tensor]
#         Dictionary with keys ``neural_input``, ``behavior_input``, ``session_id``
#         and ``subject_id``.
#     """
#     num_timesteps = int(sampling_rate * sampling_window_ms / 1000)
    
#     # ------------------------------------------------------------------
#     # 1. Bin + smooth spikes
#     # ------------------------------------------------------------------
#     smoothed_spikes = data.smoothed_spikes.smoothed_spikes

#     # ------------------------------------------------------------------
#     # 2. Prepare behaviour signal (cursor velocity)
#     # ------------------------------------------------------------------
#     behavior_input = data.cursor.vel  # np.ndarray, (timesteps?, features)


#     # ------------------------------------------------------------------
#     # 3. Align channel dimensions based on (dataset, subject, task)
#     # ------------------------------------------------------------------
#     smoothed_spikes = _ensure_dim(smoothed_spikes, MAX_NEURAL_INPUT_DIM, axis=1)
#     behavior_input = _ensure_dim(behavior_input, MAX_BEHAVIOR_INPUT_DIM, axis=1)
#     smoothed_spikes = _ensure_dim(smoothed_spikes, num_timesteps, axis=0)
#     behavior_input = _ensure_dim(behavior_input, num_timesteps, axis=0)

#     # ------------------------------------------------------------------
#     # 4. Pack into torch tensors
#     # ------------------------------------------------------------------
#     dataset, subject, task = parse_session_id(data.session.id)
#     group_tuple = (dataset, subject, task)
#     group_idx = DATASET_GROUP_TO_IDX[group_tuple]

#     return {
#         "neural_input": torch.as_tensor(smoothed_spikes, dtype=torch.float32),
#         "behavior_input": torch.as_tensor(behavior_input, dtype=torch.float32),
#         "dataset_group_idx": torch.as_tensor(group_idx, dtype=torch.int32),
#     }


def get_brainset_train_val_loaders(
    recording_id=None,
    train_config=None,
    val_config=None,
    train_batch_size=32,
    val_batch_size=32,
    seed=0,
    root=DATA_ROOT,
    transform_fn=transform_brainsets_regular_time_series_smoothed,
    num_workers=4,
    train_window_length=3.0,
    val_window_length=5.0,
    sampling_rate=200,
    keep_files_open=False,
    lazy=False,
    drop_short=True,
):
    """Sets up train and validation Datasets, Samplers, and DataLoaders
    """
    # -- Train --
    train_dataset = TorchBrainDataset(
        root=root,                # root directory where .h5 files are found
        recording_id=recording_id,  # you either specify a single recording ID
        config=train_config,                 # or a config for multi-session training / more complex configs
        split="train",
        keep_files_open=keep_files_open,
        lazy=lazy,
    )
    # We use a random sampler to improve generalization during training
    train_sampling_intervals = train_dataset.get_sampling_intervals()
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=train_window_length,
        drop_short=drop_short,
    )
    # Finally combine them in a dataloader
    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        dataset=train_dataset,      # dataset
        sampler=train_sampler,      # sampler
        batch_size=train_batch_size,
        collate_fn=partial(pad_collate, fixed_seq_len=int(train_window_length*sampling_rate)),         # the collator
        num_workers=num_workers,              # data sample processing (slicing, transforms, tokenization) happens in parallel; this sets the amount of that parallelization
        pin_memory=True,
        persistent_workers=persistent_workers
    )

    # -- Validation --
    if val_config is None:
        val_config = train_config  # if no validation config is provided, use the training config
    
    val_dataset = TorchBrainDataset(
        root=root,
        recording_id=recording_id,
        config=val_config,
        split="val",
        keep_files_open=keep_files_open,
        lazy=lazy,
    )
    # For validation we don't randomize samples for reproducibility
    val_sampling_intervals = val_dataset.get_sampling_intervals()
    val_sampler = TrialSampler(
        sampling_intervals=val_sampling_intervals
    )
    
    # Combine them in a dataloader
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=val_batch_size,
        collate_fn=partial(pad_collate, fixed_seq_len=int(val_window_length*sampling_rate)),
        num_workers=4,
        pin_memory=True,
    )

    # train_dataset.disable_data_leakage_check()
    # val_dataset.disable_data_leakage_check()
    train_dataset.transform = transform_fn
    val_dataset.transform = transform_fn

    return train_dataset, train_loader, val_dataset, val_loader

def pytorch_to_tf_generator(dataset, sampler, transform_fn):
    """
    A generator that wraps your PyTorch Dataset and a sampler.

    Args:
        dataset_params: Parameters for TorchBrainDataset
        sampler_params: Parameters for the sampler
        transform_fn: Transform function to apply to data samples

    Yields:
        Tuple of (neural_input, behavior_input, dataset_group_idx) as numpy arrays (single sample)
    """
    dataset.transform = transform_fn
    for index in sampler:
        data_sample = dataset.__getitem__(index)
        neural_input = data_sample['neural_input'].numpy()
        behavior_input = data_sample['behavior_input'].numpy()
        dataset_group_idx = data_sample['dataset_group_idx'].numpy()
        yield neural_input, behavior_input, dataset_group_idx




class NLBDictDataset(torch.utils.data.Dataset):
    def __init__(self, spikes, behavior, held_out_flags):
        self.spikes = spikes
        self.behavior = behavior
        self.held_out_flags = held_out_flags
    def __len__(self):
        return len(self.spikes)
    def __getitem__(self, i):
        return {
            'neural_input': self.spikes[i],
            'behavior_input': self.behavior[i],
            'held_out': self.held_out_flags[i]
        }

# Unified function to get held_out flags for each trial (0 for held-in, 1 for held-out)
def get_held_out_flags(trial_info, dataset, task=None):
    if dataset == 'mc_maze' and task == 'center_out_reaching':
        heldin_types = set(MC_MAZE_CONFIG.CENTER_OUT_HELD_IN_TRIAL_TYPES)
        return [0 if t in heldin_types else 1 for t in trial_info['trial_type']]
    else:
        return [0] * len(trial_info)

def get_nlb_train_val_loaders(
    dataset='mc_rtt',
    task=None,
    holdout_angles=False,
    batch_size=256,
    data_root=NLB_DATA_ROOT,
    num_workers=8,\
    
):
    """
    Loads NLB-processed data and returns train/val datasets and DataLoaders with batches matching get_train_val_loaders.

    Args:
        dataset (str): Dataset name.
        task (str): Task name. Some datasets have multiple tasks, e.g. mc_maze
        processed_data_path (str): Path to the .h5 file with NLB data.
        trial_info_path (str): Path to the .csv file with trial split info.
        batch_size (int): Batch size for DataLoaders.
        group_key (tuple): (dataset, subject, task) tuple for DATASET_GROUP_TO_IDX.
        collate_fn (callable): Collate function for DataLoader.
        device (str or torch.device, optional): If set, moves tensors to device.

    Returns:
        train_dataset, train_loader, val_dataset, val_loader
    """
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
    collate_fn = jax_collate_fn
    task_config = NLB_CONFIGS[dataset]
    
    data_path = os.path.join(data_root, task_config.H5_FILE_NAME)
    trial_info_path = os.path.join(data_root, task_config.TRIAL_INFO_FILE_NAME)
    with h5py.File(data_path, 'r') as h5file:
        dataset_dict = h5_to_dict(h5file)
    trial_info = pd.read_csv(trial_info_path)
    trial_info = trial_info[trial_info['split'].isin(['train','val'])]
    
    if task == 'center_out_reaching':
        trial_info = trial_info[trial_info['trial_version']==MC_MAZE_CONFIG.TASK_TO_TRIAL_VERSION[task]]

    if holdout_angles:
        # Use held_out flags to filter trial_info for training
        held_out_flags = get_held_out_flags(trial_info, dataset, task)
        # Only keep held-in trials for training (held_out == 0)
        train_mask = (trial_info['split'] != 'train') | (np.array(held_out_flags) == 0)
        trial_info = trial_info[train_mask]

    min_idx = trial_info['trial_id'].min()
    trial_info['trial_id'] = trial_info['trial_id'] - min_idx

    # Concatenate heldin and heldout spikes
    spikes = np.concatenate([
        dataset_dict['train_spikes_heldin'],
        dataset_dict['train_spikes_heldout']], axis=2)
    
    # Use bin_size_ms=5 to match NLB binning
    smoothed_spikes = smooth_spikes(spikes, kern_sd_ms=20, bin_size_ms=5)
    behavior = dataset_dict['train_behavior']
    # smoothed_spikes = _ensure_dim(smoothed_spikes, MAX_NEURAL_INPUT_DIM, axis=2)
    behavior = _ensure_dim(behavior, MAX_BEHAVIOR_DIM, axis=2)

    trial_info = trial_info.sort_values('trial_id').reset_index(drop=True)
    train_mask = trial_info[trial_info['split'] == 'train'].index
    val_mask = trial_info[trial_info['split'] == 'val'].index

    train_spikes = smoothed_spikes[train_mask]
    train_behavior = behavior[train_mask]
    val_spikes = smoothed_spikes[val_mask]
    val_behavior = behavior[val_mask]


    train_held_out = get_held_out_flags(trial_info[trial_info['split'] == 'train'], dataset, task)
    val_held_out = get_held_out_flags(trial_info[trial_info['split'] == 'val'], dataset, task)

    train_dataset = NLBDictDataset(train_spikes, train_behavior, train_held_out)
    val_dataset = NLBDictDataset(val_spikes, val_behavior, val_held_out)

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(pad_collate, fixed_seq_len=600),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(pad_collate, fixed_seq_len=1000),
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataset, train_loader, val_dataset, val_loader
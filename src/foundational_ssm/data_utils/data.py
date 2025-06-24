import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch_brain.data import Dataset, collate, chain
from foundational_ssm.data_utils.grouped_sampler import GroupedRandomFixedWindowSampler
from torch_brain.data.sampler import SequentialFixedWindowSampler
import h5py

# data_root = "/nfs/ghome/live/mlaimon/data/foundational_ssm/motor/processed/"
data_root = "/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/processed/"

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

def get_train_val_loaders(root=data_root, recording_id=None, train_config=None, val_config=None, batch_size=32, seed=0):
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
        batch_size=128,
        generator=torch.Generator().manual_seed(42)
    )
    # Finally combine them in a dataloader
    train_loader = DataLoader(
        dataset=train_dataset,      # dataset
        sampler=train_sampler,      # sampler
        batch_size=batch_size,      # num of samples per batch
        collate_fn=collate,         # the collator
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
    val_sampler = SequentialFixedWindowSampler(
        sampling_intervals=val_sampling_intervals,
        window_length=1.0,
    )
    # Combine them in a dataloader
    val_loader = DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=4,
        pin_memory=True,
    )

    train_dataset.disable_data_leakage_check()
    val_dataset.disable_data_leakage_check()

    return train_dataset, train_loader, val_dataset, val_loader
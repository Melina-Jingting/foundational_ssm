import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_brain.data import Dataset, collate
from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_root = "/nfs/ghome/live/mlaimon/data/foundational_ssm/motor/processed/"


def generate_sinusoidal_position_embs(num_timesteps, dim):
    position = torch.arange(num_timesteps).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe = torch.empty(num_timesteps, dim)
    pe[:, 0:dim // 2] = torch.sin(position * div_term)
    pe[:, dim//2:] = torch.cos(position * div_term)
    return pe


def load_pretrained(ckpt_path, model):
    print("Loading pretrained model...")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # poyo is pretrained using lightning, so model weights are prefixed with "model."
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    print("Done!")
    return model


def reinit_vocab(emb_module, vocab):
    emb_module.extend_vocab(vocab)
    emb_module.subset_vocab(vocab)

def move_to_gpu(data, device):
    """
    Recursively moves tensors (or collections of tensors) to the given device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_gpu(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_gpu(elem, device) for elem in data]
    else:
        return data
    


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

def custom_collate(batch):
    """
    Custom collate function to handle variable-length sequences.
    This function will pad or trim sequences to ensure they have consistent length.
    """
    
    neural_inputs = [item['neural_input'] for item in batch]
    behavior_inputs = [item['behavior_input'] for item in batch]
    neural_shapes = [x.shape[0] for x in neural_inputs]
    behavior_shapes = [x.shape[0] for x in behavior_inputs]
    
    # Find the minimum length across all sequences and trim all sequences to this length
    min_length = min(min(neural_shapes), min(behavior_shapes))
    trimmed_batch = []
    for item in batch:
        trimmed_item = {
            'neural_input': item['neural_input'][:min_length],
            'behavior_input': item['behavior_input'][:min_length],
            'session_id': item['session_id'],
            'subject_id': item['subject_id']
        }
        if 'neural_target' in item:
            trimmed_item['neural_target'] = item['neural_target'][:min_length]
        
        trimmed_batch.append(trimmed_item)
    
    final_batch = {
        'neural_input': torch.stack([item['neural_input'] for item in trimmed_batch]),
        'behavior_input': torch.stack([item['behavior_input'] for item in trimmed_batch]),
        'session_id': [item['session_id'] for item in trimmed_batch],
        'subject_id': [item['subject_id'] for item in trimmed_batch]
    }
    
    if 'neural_target' in trimmed_batch[0]:
        final_batch['neural_target'] = torch.stack([item['neural_target'] for item in trimmed_batch])
    
    return final_batch


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
    train_sampler = RandomFixedWindowSampler(
        sampling_intervals=train_sampling_intervals,
        window_length=1.0,          # context window of samples
        generator=torch.Generator().manual_seed(seed),
    )
    # Finally combine them in a dataloader
    train_loader = DataLoader(
        dataset=train_dataset,      # dataset
        sampler=train_sampler,      # sampler
        batch_size=batch_size,      # num of samples per batch
        collate_fn=custom_collate,         # the collator
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
        collate_fn=custom_collate,
        num_workers=4,
        pin_memory=True,
    )

    train_dataset.disable_data_leakage_check()
    val_dataset.disable_data_leakage_check()

    return train_dataset, train_loader, val_dataset, val_loader

import torch
import numpy as np  
from omegaconf import OmegaConf



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

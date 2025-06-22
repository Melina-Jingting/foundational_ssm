import os

import h5py
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import wandb
from omegaconf import OmegaConf

from foundational_ssm.models import S4DNeuroModel
from foundational_ssm.utils import h5_to_dict, generate_and_save_activations_wandb
from foundational_ssm.trainer import train_decoding
from foundational_ssm.data_preprocessing import smooth_spikes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processed_data_folder = '/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/processed/nlb' 
dataset_name = "mc_maze"
processed_data_path = os.path.join(processed_data_folder,dataset_name + ".h5")
trial_info_path = os.path.join(processed_data_folder,dataset_name + ".csv")

conf = {
    'task':'decoding',
    'dataset': {
        'dataset': dataset_name
    },
    'model': {
        'input_dim': 182,
        'output_dim': 2,
        'd_state': 64,
        'num_layers': 1,
        'hidden_dim': 64,
        'dropout': 0.1,
        'ssm_core':'s4d'
    },
    'optimizer': {
        'lr': 0.0005,
        'weight_decay': 0.01  # Added common parameter
    },
    'training': {
        'batch_size': 64,
        'epochs': 2000
    },
    'device': 'cuda'
}

args = OmegaConf.create(conf)

with h5py.File(processed_data_path, 'r') as h5file:
    dataset_dict = h5_to_dict(h5file)

trial_info = pd.read_csv(trial_info_path)
trial_info = trial_info[trial_info['split'].isin(['train','val'])]
min_idx = trial_info['trial_id'].min()
trial_info['trial_id'] = trial_info['trial_id'] - min_idx

train_ids = trial_info[trial_info['split']=='train']['trial_id'].tolist()
val_ids = trial_info[trial_info['split']=='val']['trial_id'].tolist()

# Concatenate both heldin and heldout spikes since we're using spikes to predict behavior
spikes = np.concat([
    dataset_dict['train_spikes_heldin'], 
    dataset_dict['train_spikes_heldout']],axis=2) 
smoothed_spikes = torch.tensor(
    smooth_spikes(spikes, kern_sd_ms=40, bin_width=5), 
    dtype=torch.float64).to(device)
behavior = torch.tensor(
    dataset_dict['train_behavior'],
    dtype=torch.float64).to(device)

input_dim = smoothed_spikes.shape[2]
output_dim = behavior.shape[2]

# Split train and val based on splits from nlb
train_dataset = TensorDataset(smoothed_spikes[train_ids], behavior[train_ids])
val_dataset = TensorDataset(smoothed_spikes[val_ids], behavior[val_ids])
full_dataset = TensorDataset(smoothed_spikes, behavior)

run_name = f"nlb_{args.task}_{args.model.ssm_core}_l{args.model.num_layers}_d{args.model.d_state}"


model = S4DNeuroModel(
    input_dim=input_dim, 
    output_dim=output_dim, 
    hidden_dim=args.model.hidden_dim,
    num_layers=args.model.num_layers,
    d_state=args.model.d_state,
    dropout=args.model.dropout
)
model.to(args.device)
model.to(torch.float64)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=args.optimizer.lr,
    weight_decay=args.optimizer.weight_decay
)
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.training.batch_size, 
    shuffle=True
)
loss_fn = nn.MSELoss()

wandb.init(
    project="foundational_ssm_nlb",
    name=run_name,
    config=conf 
)
wandb_run_id = wandb.run.id
# wandb.watch(model, log="all", log_freq=100)
train_decoding(model, 
                train_loader, 
                train_dataset.tensors, 
                val_dataset.tensors, 
                optimizer, 
                loss_fn, 
                num_epochs=args.training.epochs, 
                wandb_run_name=run_name,
                model_metadata=dict(args.model))

generate_and_save_activations_wandb(model, full_dataset.tensors, run_name)



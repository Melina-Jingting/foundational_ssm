from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors,
)
from nlb_tools.evaluation import evaluate

import numpy as np
import pandas as pd
import h5py
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from LRU_pytorch import LRU
import os
import argparse
from datetime import datetime
import wandb

import sys
# sys.path.append('/nfs/ghome/live/mlaimon/foundational_ssm/')  # Add your project root
from models import SSMNeuroModel
from losses import CombinedLoss
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
import scipy.signal as signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# Custom NLB Dataset Class
# ============================================

class NLBDataset(torch.utils.data.Dataset):
    def __init__(self, spikes_heldin, spikes_heldout=None, is_eval=False):
        """
        Args:
            spikes_heldin: Held-in neuron spike data [n_trials, time_steps, n_heldin]
            spikes_heldout: Held-out neuron spike data [n_trials, time_steps, n_heldout]
            is_eval: Whether this is evaluation data (no heldout available)
        """
        self.spikes_heldin = torch.FloatTensor(spikes_heldin)
        self.is_eval = is_eval
        
        if not is_eval and spikes_heldout is not None:
            self.spikes_heldout = torch.FloatTensor(spikes_heldout)
        else:
            self.spikes_heldout = None
            
        # Create dummy behavioral data (zeros) since our model expects it
        self.behavior = torch.zeros((spikes_heldin.shape[0], spikes_heldin.shape[1], 2))
        
        # Create dummy session and subject IDs
        self.session_ids = ['nlb_maze'] * spikes_heldin.shape[0]
        self.subject_ids = ['maze'] * spikes_heldin.shape[0]
        
    def __len__(self):
        return self.spikes_heldin.shape[0]
    
    def __getitem__(self, idx):
        item = {
            'neural_input': self.spikes_heldin[idx],
            'behavior_input': self.behavior[idx],
            'session_id': self.session_ids[idx],
            'subject_id': self.subject_ids[idx]
        }
        
        if not self.is_eval and self.spikes_heldout is not None:
            item['neural_target'] = self.spikes_heldout[idx]
            
        return item
    
    # Add these helper methods to make the model work with this dataset
    def get_session_ids(self):
        return list(set(self.session_ids))
    
    def get_subject_ids(self):
        return list(set(self.subject_ids))
    
    def get_unit_ids(self):
        # Create dummy unit IDs
        return [f'heldin{i}' for i in range(self.spikes_heldin.shape[2])]
 
# Training loop with wandb support
def train_model(model, train_loader, optimizer, loss_fn, num_epochs, use_wandb=False):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(
                neural_input=batch['neural_input'],
                behavior_input=batch['behavior_input'],
                session_id=batch['session_id'],
                subject_id=batch['subject_id']
            )
            
            # Compute loss
            loss = loss_fn(predictions['pred_neural'], batch['neural_target'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Log metrics to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss
            })

def generate_predictions(model, dataloader):
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            predictions = model(
                neural_input=batch['neural_input'],
                behavior_input=batch['behavior_input'],
                session_id=batch['session_id'],
                subject_id=batch['subject_id']
            )
            
            # Store predictions
            all_predictions.append(predictions['pred_neural'].cpu().numpy())
    
    # Concatenate predictions from all batches
    return np.concatenate(all_predictions, axis=0)

def evaluate_model(target_dict, output_dict, use_wandb=False):
    """Evaluate model and log results to wandb if enabled"""
    eval_results = evaluate(target_dict, output_dict)
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"{key}: {value}")
    
    if use_wandb:
        # Log evaluation metrics to wandb
        for metric_name, metric_value in eval_results.items():
            wandb.log({f"eval_{metric_name}": metric_value})
    
    return eval_results

def parse_args():
    parser = argparse.ArgumentParser(description="Train SSM Neural Model on NLB data")
    parser.add_argument("--dataset", type=str, default="mc_maze_small", help="NLB dataset name")
    parser.add_argument("--phase", type=str, default="val", choices=["val", "test"], help="Evaluation phase")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=128, help="SSM hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of SSM layers")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--output_dir", type=str, default="models/nlb", help="Directory to save model")
    parser.add_argument("--data_path", type=str, default="~/data/foundational_ssm/motor/raw/000128/sub-Jenkins/", 
                        help="Path to NLB dataset")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup as per NLB Tutorial
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if args.wandb:
        run_name = f"nlb_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="foundational_ssm_nlb",
            name=run_name,
            config={
                "dataset": args.dataset,
                "phase": args.phase,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "device": str(device)
            }
        )
    
    # Load dataset
    dataset_name = args.dataset
    datapath = args.data_path
    dataset = NWBDataset(datapath)
    
    # Phase and binning
    phase = args.phase
    bin_width = 5
    dataset.resample(bin_width)
    suffix = '' if bin_width == 5 else f'_{int(bin_width)}'
    
    # Prepare training data
    train_split = 'train' if phase == 'val' else ['train', 'val']
    train_dict = make_train_input_tensors(
        dataset, dataset_name=dataset_name, trial_split=train_split, save_file=False
    )
    train_spikes_heldin = train_dict['train_spikes_heldin']
    train_spikes_heldout = train_dict['train_spikes_heldout']
    print("Train held-in shape:", train_spikes_heldin.shape)
    
    # Prepare evaluation data
    eval_dict = make_eval_input_tensors(
        dataset, dataset_name=dataset_name, trial_split=phase, save_file=False
    )
    eval_spikes_heldin = eval_dict['eval_spikes_heldin']
    print("Eval dict keys:", eval_dict.keys())
    print("Eval held-in shape:", eval_spikes_heldin.shape)
    
    # Task variables
    tlength = train_spikes_heldin.shape[1]
    num_train = train_spikes_heldin.shape[0]
    num_eval = eval_spikes_heldin.shape[0]
    num_heldin = train_spikes_heldin.shape[2]
    num_heldout = train_spikes_heldout.shape[2]
    
    # Smooth spikes with 40 ms std gaussian
    kern_sd_ms = 40
    kern_sd = int(round(kern_sd_ms / dataset.bin_width))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    
    train_spksmth_heldin = np.apply_along_axis(filt, 1, train_spikes_heldin)
    eval_spksmth_heldin = np.apply_along_axis(filt, 1, eval_spikes_heldin)
    
    # Create datasets and loaders
    train_dataset = NLBDataset(train_spikes_heldin, train_spikes_heldout)
    eval_dataset = NLBDataset(eval_spikes_heldin, is_eval=True)
    
    # Create data loaders with configurable batch size
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    # Configure model with command-line args
    config = OmegaConf.create({
        "model": {
            "num_neural_features": num_heldin,
            "num_behavior_features": 2,  # Dummy behavior features
            "num_context_features": 32,
            "embedding_dim": 64,
            "ssm_projection_dim": 64,
            "ssm_hidden_dim": args.hidden_dim,
            "ssm_num_layers": args.num_layers,
            "ssm_dropout": 0.1,
            "pred_neural_dim": num_heldout,  # Predict held-out neurons
            "pred_behavior_dim": 2,  # Dummy behavior output
            "sequence_length": float(tlength / 100),  # Convert to seconds based on 100Hz
            "sampling_rate": 100,  # Assuming 100Hz sampling
            "lin_dropout": 0.1,
            "activation_fn": "relu"
        },
        "training": {
            "learning_rate": args.lr,
            "mask_prob": 0.0,  # No masking for this task
            "num_epochs": args.epochs,
            "neural_weight": 1.0,
            "behavior_weight": 0.0  # We don't care about behavior for this task
        }
    })
    
    # Initialize model
    model = SSMNeuroModel(
        num_neural_features=config.model.num_neural_features,
        num_behavior_features=config.model.num_behavior_features,
        num_context_features=config.model.num_context_features,
        embedding_dim=config.model.embedding_dim,
        ssm_projection_dim=config.model.ssm_projection_dim,
        ssm_hidden_dim=config.model.ssm_hidden_dim,
        ssm_num_layers=config.model.ssm_num_layers,
        ssm_dropout=config.model.ssm_dropout,
        pred_neural_dim=config.model.pred_neural_dim,
        pred_behavior_dim=config.model.pred_behavior_dim,
        sequence_length=config.model.sequence_length,
        sampling_rate=config.model.sampling_rate,
        lin_dropout=config.model.lin_dropout,
        activation_fn=config.model.activation_fn,
        subject_ids=train_dataset.get_subject_ids()
    )
    model = model.to(device)
    
    # Initialize vocabularies
    model.session_emb.initialize_vocab(train_dataset.get_session_ids())
    model.unit_emb.initialize_vocab(train_dataset.get_unit_ids())
    
    # Modify the model's decoder_neural_modules to ensure positive outputs (for rates)
    for subj_id in model.subject_ids:
        model.decoder_neural_modules[subj_id] = nn.Sequential(
            model.decoder_neural_modules[subj_id],
            nn.Softplus()  # Ensure non-negative rates for Poisson
        )
    
    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=False, reduction='mean')
    
    # Train the model
    train_model(model, train_loader, optimizer, loss_fn, config.training.num_epochs, use_wandb=args.wandb)
    
    # Generate predictions for train and eval datasets
    train_predictions = generate_predictions(model, DataLoader(train_dataset, batch_size=batch_size))
    eval_predictions = generate_predictions(model, eval_loader)
    
    # Reshape predictions to match the expected format
    train_rates_heldin = train_spksmth_heldin  # Keep the same smoothed held-in rates
    train_rates_heldout = train_predictions  # Our model's predictions for held-out neurons
    eval_rates_heldin = eval_spksmth_heldin  # Keep the same smoothed held-in rates
    eval_rates_heldout = eval_predictions  # Our model's predictions for held-out neurons
    
    # Prepare submission data - same format as original notebook
    output_dict = {
        dataset_name + suffix: {
            'train_rates_heldin': train_rates_heldin,
            'train_rates_heldout': train_rates_heldout,
            'eval_rates_heldin': eval_rates_heldin,
            'eval_rates_heldout': eval_rates_heldout
        }
    }
    
    # Evaluate predictions if in validation phase
    if phase == 'val':
        # Note that the RTT task is not well suited to trial averaging, so PSTHs are not made for it
        target_dict = make_eval_target_tensors(
            dataset, 
            dataset_name=dataset_name, 
            train_trial_split='train', 
            eval_trial_split='val', 
            include_psth=True, 
            save_file=False
        )
        eval_results = evaluate_model(target_dict, output_dict, use_wandb=args.wandb)

    # Save the model if output directory is provided
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"nlb_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        if args.wandb:
            wandb.save(model_path)
    
    # Close wandb
    if args.wandb:
        wandb.finish()
    
    return model, eval_results if phase == 'val' else None

if __name__ == "__main__":
    main()
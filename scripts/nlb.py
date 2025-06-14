# Standard library imports
import sys
import math
from datetime import datetime

# Third-party imports
import numpy as np
import h5py
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat
import wandb
from omegaconf import OmegaConf

# nlb_tools imports
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors,
    save_to_h5,
)
from nlb_tools.evaluation import evaluate

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y 



class S4DNeuroModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, d_state=64, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Stack of S4D layers
        self.ssm_block = nn.Sequential(
            *[S4D(hidden_dim, d_state=d_state, dropout=dropout) for _ in range(num_layers)]
        )
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, neural_input, behavior_input=None, session_id=None, subject_id=None):
        x = neural_input.transpose(1, 2)
        
        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2)
        
        x = self.ssm_block(x)
        
        # Final projection [batch, hidden, time] -> [batch, time, output_channels]
        x = x.transpose(1, 2)
        x = self.output_projection(x)
        
        return x
    

def save_model_as_artifact(model, model_name, run):
    """Save model as wandb artifact."""
    # Save model locally
    model_path = f"{model_name}.pt"
    torch.save(model.state_dict(), model_path)
    
    # Create artifact and add file
    model_artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description=f"S4D neural model for {args.dataset_name}"
    )
    model_artifact.add_file(model_path)
    
    # Log artifact to wandb
    run.log_artifact(model_artifact)
    return model_path

def generate_and_save_predictions(model, data_tensor, name, run):
    """Generate predictions and save as wandb artifact."""
    model.eval()
    with torch.no_grad():
        predictions = model(data_tensor.to(device)).cpu().numpy()
    
    # Save predictions locally
    pred_path = f"{name}_predictions.npy"
    np.save(pred_path, predictions)
    
    # Create artifact and add file
    pred_artifact = wandb.Artifact(
        name=f"{name}_predictions", 
        type="predictions",
        description=f"Model predictions on {name} data"
    )
    pred_artifact.add_file(pred_path)
    
    # Log artifact to wandb
    run.log_artifact(pred_artifact)
    return predictions, pred_path

# Training loop with wandb support
def train_model(model, train_loader, optimizer, loss_fn, num_epochs, val_spikes, val_behavior, train_spksmth_heldin_tensor, train_behavior, use_wandb=False):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            input, target = batch[0].to(device), batch[1].to(device)
            pred = model(input)
            loss = loss_fn(pred, target)

            # Forward pass + gradient descent
            optimizer.zero_grad()
            pred = model(input)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        if use_wandb:
          wandb.log({
                  "epoch": epoch + 1,
                  "train.loss": avg_epoch_loss
              })

        # Log validation loss
        if epoch % 50 == 0:
            val_pred = model(val_spikes.to(device))
            val_pred = val_pred.reshape(-1,2).cpu().detach().numpy()
            val_behavior = val_behavior.reshape(-1,2)
            val_r2 = r2_score(val_pred, val_behavior)

            train_pred = model(train_spksmth_heldin_tensor.to(device))
            train_pred = train_pred.reshape(-1,2).cpu().detach().numpy()
            train_behavior = train_behavior.reshape(-1,2)
            train_r2 = r2_score(train_pred, train_behavior)
            if use_wandb:
              wandb.log({
                  "epoch": epoch + 1,
                  "val.r2": val_r2,
                  "train.r2":train_r2
              })
            print("epoch:" + str(epoch) + " train loss: " + str(avg_epoch_loss) + " val r2: " + str(val_r2) + " train r2: "+ str(train_r2))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Load dataset
dataset_name = 'mc_maze_large'
datapath = './000138/sub-Jenkins/'
prefix = f'*ses-large'
dataset = NWBDataset(datapath, prefix)

## Dataset preparation

# Choose the phase here, either 'val' or 'test'
phase = 'val'
bin_width = 5
dataset.resample(bin_width)
suffix = '' if (bin_width == 5) else f'_{int(round(bin_width))}'

# Generate input tensors
train_trial_split = 'train' if (phase == 'val') else ['train', 'val']
train_dict = make_train_input_tensors(dataset, dataset_name=dataset_name, trial_split=train_trial_split, save_file=False, include_forward_pred=True, include_behavior=True)


## Make train input data
# Unpack input data
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
train_behavior = train_dict['train_behavior']

## Make eval input data
eval_trial_split = phase
eval_dict = make_eval_input_tensors(dataset, dataset_name=dataset_name, trial_split=eval_trial_split, save_file=False)
eval_spikes_heldin = eval_dict['eval_spikes_heldin']


## Prep input
# Combine train spiking data into one array
train_spikes_heldin = train_dict['train_spikes_heldin']
train_spikes_heldout = train_dict['train_spikes_heldout']
train_spikes_heldin_fp = train_dict['train_spikes_heldin_forward']
train_spikes_heldout_fp = train_dict['train_spikes_heldout_forward']
train_spikes = np.concatenate([
    np.concatenate([train_spikes_heldin, train_spikes_heldin_fp], axis=1),
    np.concatenate([train_spikes_heldout, train_spikes_heldout_fp], axis=1),
], axis=2)

# Fill missing test spiking data with zeros and make masks
eval_spikes_heldin = eval_dict['eval_spikes_heldin']
eval_spikes = np.full((eval_spikes_heldin.shape[0], train_spikes.shape[1], train_spikes.shape[2]), 0.0)
masks = np.full((eval_spikes_heldin.shape[0], train_spikes.shape[1], train_spikes.shape[2]), False)
eval_spikes[:, :eval_spikes_heldin.shape[1], :eval_spikes_heldin.shape[2]] = eval_spikes_heldin
masks[:, :eval_spikes_heldin.shape[1], :eval_spikes_heldin.shape[2]] = True

# Make lists of arrays
train_datas = [train_spikes[i, :, :].astype(int) for i in range(len(train_spikes))]
eval_datas = [eval_spikes[i, :, :].astype(int) for i in range(len(eval_spikes))]
eval_masks = [masks[i, :, :].astype(bool) for i in range(len(masks))]

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

# Data dimensions
input_dim = train_spksmth_heldin.shape[2]
output_dim = train_behavior.shape[2]


# Create dataset and split 
train_spksmth_heldin_tensor = torch.tensor(train_spksmth_heldin, dtype=torch.float32)
eval_spksmth_heldin_tensor = torch.tensor(eval_spksmth_heldin, dtype=torch.float32)
train_behavior_tensor = torch.tensor(train_behavior, dtype=torch.float32)

tensor_dataset = TensorDataset(train_spksmth_heldin_tensor, train_behavior_tensor)
rng = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(tensor_dataset, [0.9,0.1], generator=rng)
val_spikes, val_behavior = tensor_dataset[val_dataset.indices]


conf = {
    'dataset_name': dataset_name,
    'task': 'decoding',

    'batch_size': 64,
    'epochs': 2000,
    'lr': 0.0005,

    'num_layers': 2,
    'hidden_dim': 64,
    'dropout': 0.1,
    'd_state': 64,

    'device':'cuda'
}

args = OmegaConf.create(conf)

run_name = f"nlb_{args.dataset_name}_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
wandb.init(
    project="foundational_ssm_nlb",
    name=run_name,
    config={
        "dataset": args.dataset_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "device": args.device
    }
)

model = S4DNeuroModel(input_dim = input_dim, output_dim=output_dim, num_layers=args.num_layers)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
loss_fn = nn.MSELoss()


train_model(model, train_loader, optimizer, loss_fn, args.epochs, val_spikes, val_behavior, train_spksmth_heldin_tensor, train_behavior, use_wandb=True)
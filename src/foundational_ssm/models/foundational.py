import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

# Import model components
from .s4d import S4D

# Import data processing utilities
from foundational_ssm.data_utils import bin_spikes, map_binned_features_to_global
from foundational_ssm.constants import DATASET_GROUP_DIMS, DATASET_GROUPS, DATASET_GROUP_TO_IDX

from torch_brain.nn import InfiniteVocabEmbedding
from temporaldata import Data

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr
from typing import List, Dict
from .s5 import S5Block


class S4DNeuroModel(nn.Module):
    def __init__(self, 
        input_dim, 
        output_dim, 
        d_state=64, 
        num_layers=2, 
        hidden_dim=64, 
        dropout=0.1,
        ssm_core: str = "s4d"):
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


class SSMFoundational(eqx.Module):
    context_embedding: eqx.nn.Embedding
    encoders: List[eqx.nn.Linear]          # group_key → encoder
    ssm_blocks: List[S5Block]
    decoders: List[eqx.nn.Linear]
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        rng_seed,
        ssm_io_dim, # dim of input and output of the SSM, H in the S5 paper
        ssm_dim, # dim of ssm states, P in the S5 paper
        ssm_init_diag_blocks, # S5 initializes with blocks of diagonals of HiPPO matrices
        ssm_num_layers, # number of layers of SSMs
        output_dim, # dim of final output of the model
        context_dim = 8, # dim of context embedding
        C_init: str = "trunc_standard_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretisation: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0
    ):
        
        num_dataset_groups = len(DATASET_GROUP_DIMS)
        
        key = jr.PRNGKey(rng_seed)  
        encoder_key, block_key, decoder_key, embedding_key = jr.split(key, 4)
    
        self.context_embedding = eqx.nn.Embedding(num_dataset_groups, context_dim, key=embedding_key)
        
        block_keys = jr.split(block_key, ssm_init_diag_blocks)
        
        # Create encoders dict
        MAX_RAW_INPUT_DIM = max(dims[0] for dims in DATASET_GROUP_DIMS.values())
        self.encoders = [
            eqx.nn.Linear(MAX_RAW_INPUT_DIM, ssm_io_dim-context_dim, key=encoder_key)
            for group in DATASET_GROUPS
        ]
        self.decoders = [
            eqx.nn.Linear(ssm_dim, output_dim, key=decoder_key)
            for group in DATASET_GROUPS
        ]
            
        self.ssm_blocks = [
            S5Block(
                ssm_size = ssm_dim,
                blocks = ssm_init_diag_blocks,
                H = ssm_io_dim,
                C_init = C_init,
                conj_sym = conj_sym,
                clip_eigs = clip_eigs,
                discretisation = discretisation,
                dt_min = dt_min,
                dt_max = dt_max,
                step_rescale = step_rescale,
                key=key,
            )
            for key in block_keys
        ]
        

    def __call__(self, x, state, key, group_idx):
        """Compute S5 for a specific dataset. Returns output, state, and a dict of intermediate SSM block outputs."""
        # 1. Project input to SSM dimension
        dropkeys = jr.split(key, len(self.ssm_blocks))
        encoders_vmap = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
        x = jax.lax.switch(group_idx, encoders_vmap, x)
        
        # 2. Add context vector
        context_vec = self.context_embedding(group_idx) 
        broadcast_context = jnp.broadcast_to(context_vec, (x.shape[0],) + context_vec.shape)
        x = jnp.concatenate([x, broadcast_context], axis=1)
        
        # 3. Apply S5 blocks and collect activations
        activations = {}
        for i, (block, key) in enumerate(zip(self.ssm_blocks, dropkeys)):
            x, state = block(x, state, key=key)
            activations[f'ssm_block_{i}'] = x
        
        # 4. Project output to behavior dimension
        decoders_vmap = [jax.vmap(dec, in_axes=0, out_axes=0) for dec in self.decoders]
        x = jax.lax.switch(group_idx, decoders_vmap, x)
        return x, state, activations





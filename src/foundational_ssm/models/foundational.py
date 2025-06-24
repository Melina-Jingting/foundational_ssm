import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict

# Import model components
from .s4d import S4D

# Import data processing utilities
from foundational_ssm.data_utils import bin_spikes, map_binned_features_to_global
from foundational_ssm.constants.dataset_info import GROUP_DIMS

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
    encoders: Dict[str, eqx.nn.Linear]          # group_key → encoder
    ssm_blocks: List[S5Block]
    decoder: eqx.nn.Linear
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        key,
        num_blocks,
        N,
        ssm_size,
        ssm_blocks,
        H,
        output_dim,
        C_init: str = "trunc_standard_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretisation: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0
    ):

        encoder_key, block_key, decoder_key, weight_key = jr.split(key, 4)
        encoder_keys = jr.split(encoder_key, len(GROUP_DIMS))
        block_keys = jr.split(block_key, num_blocks)
        
        # Create encoders dict
        encoders_dict = {}
        for group_key, encoder_key in zip(GROUP_DIMS.keys(), encoder_keys):
            input_dim = GROUP_DIMS[group_key][0]  # neural_dim
            group_key_str = f"{group_key[0]}-{group_key[1]}-{group_key[2]}"
            encoders_dict[group_key_str] = eqx.nn.Linear(input_dim, H, key=encoder_key)
        
        self.encoders = encoders_dict
        
        self.ssm_blocks = [
            S5Block(
                ssm_size,
                ssm_blocks,
                H,
                C_init,
                conj_sym,
                clip_eigs,
                discretisation,
                dt_min,
                dt_max,
                step_rescale,
                key=key,
            )
            for key in block_keys
        ]
        self.decoder = eqx.nn.Linear(H, output_dim, key=decoder_key)

    def __call__(self, x, state, key, group_key):
        """Compute S5 for a specific dataset."""
        dropkeys = jr.split(key, len(self.ssm_blocks))
        x = jax.vmap(self.encoders[group_key])(x)
        for block, key in zip(self.ssm_blocks, dropkeys):
            x, state = block(x, state, key=key)
        x = jax.vmap(self.decoder)(x)
        return x, state





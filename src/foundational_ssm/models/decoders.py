import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


# Import data processing utilities
from foundational_ssm.constants import DATASET_GROUP_INFO, DATASET_GROUPS, MAX_NEURAL_UNITS

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr
from typing import List, Optional

from .s5 import S5Block


class SSMFoundationalDecoder(eqx.Module):
    context_embedding: eqx.nn.Embedding
    encoders: List[eqx.nn.Linear]          # group_key → encoder
    ssm_blocks: List[S5Block]
    decoder: eqx.nn.Linear
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        rng_seed,
        input_dim = MAX_NEURAL_UNITS,
        num_dataset_groups = len(DATASET_GROUP_INFO),
        ssm_io_dim = 64, # dim of input and output of the SSM, H in the S5 paper
        ssm_dim = 64, # dim of ssm states, P in the S5 paper
        ssm_init_diag_blocks = 4, # S5 initializes with blocks of diagonals of HiPPO matrices
        ssm_num_layers = 4, # number of layers of SSMs
        output_dim = 2, # dim of final output of the model
        context_dim = 8, # dim of context embedding
        C_init: str = "trunc_standard_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretisation: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0
    ):
        
        
        key = jr.PRNGKey(rng_seed)  
        encoder_key, block_key, decoder_key, embedding_key = jr.split(key, 4)
    
        self.context_embedding = eqx.nn.Embedding(num_dataset_groups, context_dim, key=embedding_key)
        
        block_keys = jr.split(block_key, ssm_num_layers)
        
        # Create encoders dict
        self.encoders = [
            eqx.nn.Linear(input_dim, ssm_io_dim-context_dim, key=encoder_key)
            for _ in range(num_dataset_groups)
        ]
        self.decoder = eqx.nn.Linear(ssm_dim, output_dim, key=decoder_key)
            
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
        

    def __call__(self, x, state, group_idx, key, inference=False):
        """Compute S5 for a specific dataset. Returns output, state, and a dict of intermediate SSM block outputs."""
        # 1. Project input to SSM dimension
        encoders_vmap = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
        x = jax.lax.switch(group_idx, encoders_vmap, x)
        
        # 2. Add context vector
        context_vec = self.context_embedding(group_idx) 
        broadcast_context = jnp.broadcast_to(context_vec, (x.shape[0],) + context_vec.shape)
        x = jnp.concatenate([x, broadcast_context], axis=1)
        
        # 3. Apply S5 blocks and collect activations
        dropkeys = jr.split(key, len(self.ssm_blocks))
        for i, (block, key) in enumerate(zip(self.ssm_blocks, dropkeys)):
            x, state = block(x, state, key=key, inference=inference)
        
        # 4. Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)
        return x, state
    
    def call_with_activations(self, x, state, group_idx):
        """Compute S5 for a specific dataset. Returns output, state, and a dict of intermediate SSM block outputs."""
        # 1. Project input to SSM dimension
        
        encoders_vmap = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
        x = jax.lax.switch(group_idx, encoders_vmap, x)
        
        # 2. Add context vector
        context_vec = self.context_embedding(group_idx) 
        broadcast_context = jnp.broadcast_to(context_vec, (x.shape[0],) + context_vec.shape)
        x = jnp.concatenate([x, broadcast_context], axis=1)
        
        # 3. Apply S5 blocks and collect activations
        activations_list = []
        key = jr.PRNGKey(0) # just for compatibility
        dropkeys = jr.split(key, len(self.ssm_blocks))
        for i, (block, key) in enumerate(zip(self.ssm_blocks, dropkeys)):
            x, state = block(x, state, key=key, inference=True)
            activations_list.append(x)
        
        # 4. Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)
        return x, activations_list, state


class SSMDownstreamDecoder(eqx.Module):
    context_embedding: jax.Array  # shape: (context_dim,)
    encoder: eqx.nn.Linear
    ssm_blocks: List[S5Block]
    decoder: eqx.nn.Linear

    def __init__(
        self,
        rng_seed,
        input_dim,
        ssm_io_dim,
        ssm_dim,
        ssm_init_diag_blocks,
        ssm_num_layers,
        output_dim,
        context_dim=8,
        pretrained_ssm_blocks: Optional[List] = None,
        pretrained_decoder: Optional[eqx.nn.Linear] = None,
        C_init: str = "trunc_standard_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretisation: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
    ):
        key = jr.PRNGKey(rng_seed)
        encoder_key, block_key, decoder_key, embedding_key = jr.split(key, 4)

        # Single context embedding vector (learnable)
        self.context_embedding = jax.random.normal(embedding_key, (context_dim,))

        # Single encoder for this task
        self.encoder = eqx.nn.Linear(input_dim, ssm_io_dim - context_dim, key=encoder_key)

        # SSM blocks: use pretrained if provided, else initialize new
        if pretrained_ssm_blocks is not None:
            self.ssm_blocks = pretrained_ssm_blocks
        else:
            block_keys = jr.split(block_key, ssm_num_layers)
            self.ssm_blocks = [
                S5Block(
                    ssm_size=ssm_dim,
                    blocks=ssm_init_diag_blocks,
                    H=ssm_io_dim,
                    C_init=C_init,
                    conj_sym=conj_sym,
                    clip_eigs=clip_eigs,
                    discretisation=discretisation,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    step_rescale=step_rescale,
                    key=key,
                )
                for key in block_keys
            ]

        # Decoder: use pretrained if provided, else initialize new
        if pretrained_decoder is not None:
            self.decoder = pretrained_decoder
        else:
            self.decoder = eqx.nn.Linear(ssm_dim, output_dim, key=decoder_key)

    def __call__(self, x, state, key, inference=False):
        # Project input to SSM dimension
        x = jax.vmap(self.encoder)(x)
        # Add context vector
        context_vec = jnp.broadcast_to(self.context_embedding, (x.shape[0],) + self.context_embedding.shape)
        x = jnp.concatenate([x, context_vec], axis=1)
        # Apply SSM blocks
        dropkeys = jr.split(key, len(self.ssm_blocks))
        for block, k in zip(self.ssm_blocks, dropkeys):
            x, state = block(x, state, key=k, inference=inference)
        # Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)
        return x, state

    def call_with_activations(self, x, state, key, inference=True):
        x = jax.vmap(self.encoder)(x)
        context_vec = jnp.broadcast_to(self.context_embedding, (x.shape[0],) + self.context_embedding.shape)
        x = jnp.concatenate([x, context_vec], axis=1)
        dropkeys = jr.split(key, len(self.ssm_blocks))
        activations_list = []
        for i, (block, k) in enumerate(zip(self.ssm_blocks, dropkeys)):
            x, state = block(x, state, key=k, inference=inference)
            activations_list.append(x)
        x = jax.vmap(self.decoder)(x)
        return x, activations_list, state

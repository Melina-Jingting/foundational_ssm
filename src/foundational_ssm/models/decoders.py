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
import jax.tree_util as jtu
from typing import List, Optional

from .s5 import S5Block, GLU


class SSMFoundationalDecoder(eqx.Module):
    context_embedding: eqx.nn.Embedding
    encoders: List[eqx.nn.Linear]          # group_key → encoder
    encoder_dropout: eqx.nn.Dropout
    ssm_blocks: List[S5Block]
    decoder: eqx.nn.Linear
    decoder_dropout: eqx.nn.Dropout
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        rng_seed,
        input_dim = MAX_NEURAL_UNITS,
        num_dataset_groups = 10,
        ssm_io_dim = 32, # dim of input and output of the SSM, H in the S5 paper
        ssm_dim = 32, # dim of ssm states, P in the S5 paper
        ssm_init_diag_blocks = 4, # S5 initializes with blocks of diagonals of HiPPO matrices
        ssm_num_layers = 2, # number of layers of SSMs
        output_dim = 2, # dim of final output of the model
        context_dim = 4, # dim of context embedding
        dropout_p: float = 0.3,
        ssm_dropout_p: float = 0.05,
        C_init: str = "trunc_standard_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretisation: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0
    ):
        
        
        key = jr.PRNGKey(rng_seed)  
        encoder_key, glu_key, block_key, decoder_key, embedding_key = jr.split(key, 5)

        self.context_embedding = eqx.nn.Embedding(num_dataset_groups, context_dim, key=embedding_key)
        self.encoder_dropout = eqx.nn.Dropout(p=dropout_p)
        
        block_keys = jr.split(block_key, ssm_num_layers)
        
        # Create encoders dict
        encoder_in_dim = ssm_io_dim - context_dim
        self.encoders = [
            eqx.nn.Linear(input_dim, encoder_in_dim, key=encoder_key)
            for _ in range(num_dataset_groups)
        ]

        # Manually rescale depending on effective dimensions in dataset group
        new_encoders = []
        for i, encoder in enumerate(self.encoders):
            scale_factor = jnp.sqrt(input_dim / encoder.weight.shape[0])
            new_encoder = jtu.tree_map(
                lambda w: w * scale_factor if w.ndim == 2 else w, 
                encoder
            )
            new_encoders.append(new_encoder)
        self.encoders = new_encoders

        self.decoder = eqx.nn.Linear(ssm_io_dim, output_dim, key=decoder_key)
        self.decoder_dropout = eqx.nn.Dropout(p=dropout_p)
            
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
                drop_rate=ssm_dropout_p
            )
            for key in block_keys
        ]
        

    def __call__(self, x, state, group_idx, key):
        """Compute S5 for a specific dataset. Returns output, state, and a dict of intermediate SSM block outputs."""
        # 1. Project input to SSM dimension
        encoders_vmap = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
        x = jax.lax.switch(group_idx, encoders_vmap, x)

        # Split key for all dropout layers
        num_dropout_layers = 2 + len(self.ssm_blocks)
        dropkeys = jr.split(key, num_dropout_layers)
        
        x = self.encoder_dropout(x, key=dropkeys[0])
        
        # 2. Add context vector
        context_vec = self.context_embedding(group_idx) 
        broadcast_context = jnp.broadcast_to(context_vec, (x.shape[0],) + context_vec.shape)
        x = jnp.concatenate([x, broadcast_context], axis=1)
        
        # 3. Apply S5 blocks and collect activations
        for i, (block, key) in enumerate(zip(self.ssm_blocks, dropkeys[1:-1])):
            x, state = block(x, state, key=key)
        
        # 4. Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)
        x = self.decoder_dropout(x, key=dropkeys[-1])
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
            x, x_pre_activation, state = block.call_with_activations(x, state, key=key)
            activations_list.append(x_pre_activation)

        # 4. Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)
        return x, activations_list, state


class SSMDownstreamDecoder(eqx.Module):
    context_embedding: jax.Array  # shape: (context_dim,)
    encoder: eqx.nn.Linear
    encoder_dropout: eqx.nn.Dropout
    ssm_blocks: List[S5Block]
    decoder: eqx.nn.Linear
    decoder_dropout: eqx.nn.Dropout

    def __init__(
        self,
        rng_seed,
        input_dim,
        ssm_io_dim,
        ssm_dim,
        ssm_init_diag_blocks,
        ssm_num_layers,
        output_dim,
        context_dim=4,
        dropout_p: float = 0.1,
        ssm_dropout_p: float = 0.05,
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
        encoder_key, glu_key, block_key, decoder_key, embedding_key = jr.split(key, 5)

        # Single context embedding vector (learnable)
        self.context_embedding = jax.random.normal(embedding_key, (context_dim,))

        # Single encoder for this task
        encoder_in_dim = ssm_io_dim - context_dim
        self.encoder = eqx.nn.Linear(input_dim, encoder_in_dim, key=encoder_key)
        self.encoder_dropout = eqx.nn.Dropout(p=dropout_p)

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
                    drop_rate=ssm_dropout_p,
                    key=key,
                )
                for key in block_keys
            ]

        # Decoder: use pretrained if provided, else initialize new
        if pretrained_decoder is not None:
            self.decoder = pretrained_decoder
        else:
            self.decoder = eqx.nn.Linear(ssm_io_dim, output_dim, key=decoder_key)
        self.decoder_dropout = eqx.nn.Dropout(p=dropout_p)

    def __call__(self, x, state, key):
        # Project input to SSM dimension
        x = jax.vmap(self.encoder)(x)

        # Split key for all dropout layers
        num_dropout_layers = 2 + len(self.ssm_blocks)
        dropkeys = jr.split(key, num_dropout_layers)

        x = self.encoder_dropout(x, key=dropkeys[0])
        # x = jax.vmap(self.glu)(x)

        # Add context vector
        context_vec = jnp.broadcast_to(self.context_embedding, (x.shape[0],) + self.context_embedding.shape)
        x = jnp.concatenate([x, context_vec], axis=1)
        # Apply SSM blocks
        for block, k in zip(self.ssm_blocks, dropkeys[1:-1]):
            x, state = block(x, state, key=k)
        # Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)
        x = self.decoder_dropout(x, key=dropkeys[-1])
        return x, state

    def call_with_activations(self, x, state, layer_keys=[]):
        """
        Computes S5 and optionally returns a dictionary of intermediate activations.
        
        Args:
            x: Input tensor
            state: Model state
            key: PRNG key
            layer_keys: List of activation keys to capture. Special case: if "ssm_post_activation"
                        is included, activations for all SSM layers will be captured.
        """
        def _capture(k, v):
            if k in layer_keys:
                activations[k] = v
                return
            
            if "ssm_post_activation" in layer_keys and k.startswith("ssm_post_activation_"):
                activations[k] = v
                return
            
        activations = {}
        layer_keys = layer_keys
        
        # 1. encode + dropout + context
        x = jax.vmap(self.encoder)(x)
        _capture("post_encoder", x)

        context_vec = jnp.broadcast_to(self.context_embedding, (x.shape[0],) + self.context_embedding.shape)
        x = jnp.concatenate([x, context_vec], axis=1)
        
        # 2. SSM blocks
        for i, block in enumerate(self.ssm_blocks):
            x, state, block_activations = block.call_with_activations(x, state, layer_keys=layer_keys)
            activations.update({f"{k}_{i}": v for k, v in block_activations.items()})
            _capture(f"ssm_post_activation_{i}", x)
        
        # 3. Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)        
        return x, state, activations

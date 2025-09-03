from typing import Literal

import jax
import jax.numpy as jnp
import equinox as eqx

# Import data processing utilities
from foundational_ssm.constants import DATASET_GROUP_INFO, DATASET_GROUPS, MAX_NEURAL_UNITS

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random as jr
from typing import List, Optional

from .s5 import S5Block
from .muP import compute_muP_scale, muP_init, make_muP_init
from .linear import Linear, default_init 
from .field import field

class SSMFoundationalDecoder(eqx.Module):
    context_embedding: eqx.nn.Embedding
    encoders: List[Linear]          # group_key → encoder
    encoder_dropout: eqx.nn.Dropout
    ssm_blocks: List[S5Block]
    decoder: Linear
    decoder_dropout: eqx.nn.Dropout
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False
    _context_dim: int | Literal["scalar"] = field(static=True)

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
        init: str = "standard", # "standard" or "muP"
        C_init: str = "trunc_standard_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretisation: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
        use_glu: bool = True, 
        effective_input_dims: Optional[List[int]] = None,
    ):
        key = jr.PRNGKey(rng_seed)
        encoder_key, block_key, decoder_key, embedding_key = jr.split(key, 4)
        self._context_dim = context_dim
        # Create embedding only if context is used
        self.context_embedding = (
            eqx.nn.Embedding(num_dataset_groups, context_dim, key=embedding_key)
            if context_dim > 0
            else None
        )     
        self.encoder_dropout = eqx.nn.Dropout(p=dropout_p)

        block_keys = jr.split(block_key, ssm_num_layers)

        # Group-specific encoders; allow effective input dim per group for muP
        encoder_in_dim = ssm_io_dim - context_dim
        self.encoders = []
        enc_keys = jr.split(encoder_key, num_dataset_groups)
        for gi in range(num_dataset_groups):
            if init == "muP":
                fan_in_eff = (
                    effective_input_dims[gi]
                    if (effective_input_dims is not None and gi < len(effective_input_dims))
                    else input_dim
                )
                enc_init = make_muP_init(fan_out_override=encoder_in_dim, fan_in_override=fan_in_eff)
                encoder = Linear(input_dim, encoder_in_dim, key=enc_keys[gi], init_fn=enc_init)
            else:
                encoder = Linear(input_dim, encoder_in_dim, key=enc_keys[gi], init_fn=default_init)
            self.encoders.append(encoder)

        # Decoder
        if init == "muP":
            dec_init = make_muP_init(fan_out_override=output_dim, fan_in_override=ssm_io_dim)
            self.decoder = Linear(ssm_io_dim, output_dim, key=decoder_key, init_fn=dec_init)
        else:
            self.decoder = Linear(ssm_io_dim, output_dim, key=decoder_key, init_fn=default_init)
        self.decoder_dropout = eqx.nn.Dropout(p=dropout_p)

        # SSM blocks
        self.ssm_blocks = [
            S5Block(
                ssm_size=ssm_dim,
                blocks=ssm_init_diag_blocks,
                H=ssm_io_dim,
                init=init,
                C_init=C_init,
                conj_sym=conj_sym,
                clip_eigs=clip_eigs,
                discretisation=discretisation,
                dt_min=dt_min,
                dt_max=dt_max,
                step_rescale=step_rescale,
                use_glu=use_glu,  # <-- Pass the parameter down
                key=k,
                drop_rate=ssm_dropout_p,
            )
            for k in block_keys
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
        if self._context_dim > 0:
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

    def call_with_activations(self, x, state, group_idx, layer_keys):

        activations = {}

        def _capture(k, v):
            if k in layer_keys:
                activations[k] = v
                return
            
        """Compute S5 for a specific dataset. Returns output, state, and a dict of intermediate SSM block outputs."""
        # 1. Project input to SSM dimension
        encoders_vmap = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
        x = jax.lax.switch(group_idx, encoders_vmap, x)
        _capture("post_encoder", x)
        
        # 2. Add context vector
        if self._context_dim > 0:
            context_vec = self.context_embedding(group_idx)
            broadcast_context = jnp.broadcast_to(context_vec, (x.shape[0],) + context_vec.shape)
            x = jnp.concatenate([x, broadcast_context], axis=1)
        
        
        # 3. Apply S5 blocks and collect activations
        key = jr.PRNGKey(0) # just for compatibility
        dropkeys = jr.split(key, len(self.ssm_blocks))
        for i, (block, key) in enumerate(zip(self.ssm_blocks, dropkeys)):
            x, state, block_activations = block.call_with_activations(x, state, layer_keys=layer_keys)
            activations.update({f"{k}_{i}": v for k, v in block_activations.items()})

        # 4. Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)
        return x, state, activations



class SSMDownstreamDecoder(eqx.Module):
    context_embedding: jax.Array  # shape: (context_dim,)
    encoder: Linear
    encoder_dropout: eqx.nn.Dropout
    ssm_blocks: List[S5Block]
    decoder: Linear
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
        init: str = "standard",
        C_init: str = "trunc_standard_normal",
        conj_sym: bool = True,
        clip_eigs: bool = False,
        discretisation: str = "zoh",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        step_rescale: float = 1.0,
        use_glu: bool = True
    ):
        key = jr.PRNGKey(rng_seed)
        encoder_key, glu_key, block_key, decoder_key, embedding_key = jr.split(key, 5)
        init_fn = default_init if init == "standard" else muP_init

        # Single context embedding vector (learnable)
        self.context_embedding = jax.random.normal(embedding_key, (context_dim,))

        # Single encoder for this task
        encoder_in_dim = ssm_io_dim - context_dim
        # if init == "muP":
        #     enc_init = make_muP_init(fan_out_override=encoder_in_dim, fan_in_override=input_dim)
        #     self.encoder = Linear(input_dim, encoder_in_dim, key=encoder_key, init_fn=enc_init)
        # else:
        self.encoder = Linear(input_dim, encoder_in_dim, key=encoder_key, init_fn=default_init)
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
                    init=init,
                    C_init=C_init,
                    conj_sym=conj_sym,
                    clip_eigs=clip_eigs,
                    discretisation=discretisation,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    step_rescale=step_rescale,
                    use_glu=use_glu,
                    drop_rate=dropout_p,
                    key=key,
                )
                for key in block_keys
            ]

        # Decoder: use pretrained if provided, else initialize new
        if init == "muP":
            dec_init = make_muP_init(fan_out_override=output_dim, fan_in_override=ssm_io_dim)
            self.decoder = Linear(ssm_io_dim, output_dim, key=decoder_key, init_fn=dec_init)
        else:
            self.decoder = Linear(ssm_io_dim, output_dim, key=decoder_key, init_fn=default_init)
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
        
        # 3. Project output to behavior dimension
        x = jax.vmap(self.decoder)(x)        
        return x, state, activations

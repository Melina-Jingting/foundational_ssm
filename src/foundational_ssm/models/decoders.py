from typing import Union, Any, Dict

import jax
import equinox as eqx

from jax import random as jr
from typing import List, Optional
from .s5 import S5Layer
from .ssm import ContinuousSSMLayer

SSM_LAYER_REGISTRY = {
    "S5Layer": S5Layer,
    "ContinuousSSMLayer": ContinuousSSMLayer,
}

class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))

class SSMBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    ssm: eqx.Module
    glu: eqx.Module  # Changed from GLU to the more general eqx.Module
    drop: eqx.nn.Dropout

    def __init__(
        self,
        dim_ssm_io,
        ssm_layer_cls,
        ssm_layer_args,
        drop_rate=0.05,
        *,
        key,
    ):
        ssmkey, glukey = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=dim_ssm_io, axis_name="batch", channelwise_affine=False, mode="batch"
        )
        
        self.ssm = ssm_layer_cls(
            dim_ssm_io=dim_ssm_io,
            **ssm_layer_args,
            key=ssmkey,
        )

        # Conditionally initialize the GLU or an Identity layer
        self.glu = GLU(dim_ssm_io, dim_ssm_io, key=glukey)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    # No changes are needed for the __call__ method.
    def __call__(self, x, state, *, key):
        """Compute S5 block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x
        x, state = self.norm(x.T, state)
        x = x.T
        x = self.ssm(x)
        x = jax.nn.gelu(x)
        x = self.drop(x, key=dropkey1)
        x = jax.vmap(self.glu)(x)  # This line works for both GLU and Identity
        x = self.drop(x, key=dropkey2)
        # x = skip + x
        return x, state

    def call_with_activations(self, x, state, layer_keys):
        """Compute S5 block."""
        activations = {}
        _capture = (
            lambda k, v: activations.update({k: v})
            if layer_keys and k in layer_keys
            else None
        )
        x, state = self.norm(x.T, state)
        x = x.T
        ssm_y, ssm_x = self.ssm.call_with_activations(x)
        _capture("ssm_x", ssm_x)
        _capture("ssm_y", ssm_y)
        post_gelu = jax.nn.gelu(ssm_y)
        _capture("ssm_post_gelu", post_gelu)
        post_glu = jax.vmap(self.glu)(post_gelu)
        _capture("ssm_post_glu", post_glu)
        return post_glu, state, activations


class _MultiEncoder(eqx.Module):
    """Helper module to handle group-specific encoders."""
    encoders: List[eqx.nn.Linear]

    def __init__(self, input_dim, output_dim, num_groups, *, key):
        keys = jr.split(key, num_groups)
        self.encoders = [eqx.nn.Linear(input_dim, output_dim, key=k) for k in keys]

    def __call__(self, x, group_idx):
        # x shape: (Length, InputDim)
        # We vmap the encoders over the sequence length
        funcs = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
        return jax.lax.switch(group_idx, funcs, x)

class SSMDecoder(eqx.Module):
    encoder: Union[eqx.nn.Linear, _MultiEncoder]
    encoder_dropout: eqx.nn.Dropout
    ssm_blocks: List[SSMBlock]
    decoder: eqx.nn.Linear
    decoder_dropout: eqx.nn.Dropout
    
    # Metadata flags
    stateful: bool = True
    nondeterministic: bool = True
    lip2: bool = False

    def __init__(
        self,
        rng_seed,
        input_dim,
        output_dim,
        dim_ssm_io=32,
        num_ssm_layers=2,
        num_dataset_groups=1,
        dropout_p=0.1,
        ssm_layer_cls: Union[str, Any] = "S5Layer",
        ssm_layer_args: Dict = {},
    ):
        key = jr.PRNGKey(rng_seed)
        encoder_key, block_key, decoder_key = jr.split(key, 3)
        
        # Resolve SSM Layer Class
        if isinstance(ssm_layer_cls, str):
            if ssm_layer_cls not in SSM_LAYER_REGISTRY:
                raise ValueError(f"Unknown SSM layer class: {ssm_layer_cls}")
            ssm_layer_cls = SSM_LAYER_REGISTRY[ssm_layer_cls]

        # Initialize Encoder (Single or Multi-Group)
        if num_dataset_groups > 1:
            self.encoder = _MultiEncoder(
                input_dim, dim_ssm_io, num_dataset_groups, key=encoder_key
            )
        else:
            self.encoder = eqx.nn.Linear(input_dim, dim_ssm_io, key=encoder_key)
            
        self.encoder_dropout = eqx.nn.Dropout(p=dropout_p)

        block_keys = jr.split(block_key, num_ssm_layers)
        self.ssm_blocks = [
            SSMBlock(
                dim_ssm_io=dim_ssm_io,
                ssm_layer_cls=ssm_layer_cls,
                ssm_layer_args=ssm_layer_args,
                drop_rate=dropout_p,
                key=k,
            )
            for k in block_keys
        ]

        self.decoder = eqx.nn.Linear(dim_ssm_io, output_dim, key=decoder_key)
        self.decoder_dropout = eqx.nn.Dropout(p=dropout_p)

    def __call__(self, x, state, key, group_idx=None):
        """
        Compute SSM Decoder.
        Args:
            x: Input tensor.
            state: Model state.
            key: PRNG key.
            group_idx: Dataset group index (required if num_dataset_groups > 1).
        """
        if isinstance(self.encoder, _MultiEncoder):
            if group_idx is None:
                raise ValueError("group_idx must be provided for multi-group encoder")
            x = self.encoder(x, group_idx)
        else:
            x = jax.vmap(self.encoder)(x)

        num_dropout_layers = 2 + len(self.ssm_blocks)
        dropkeys = jr.split(key, num_dropout_layers)

        x = self.encoder_dropout(x, key=dropkeys[0])

        for block, k in zip(self.ssm_blocks, dropkeys[1:-1]):
            x, state = block(x, state, key=k)

        x = jax.vmap(self.decoder)(x)
        x = self.decoder_dropout(x, key=dropkeys[-1])
        return x, state

    def call_with_activations(self, x, state, layer_keys, group_idx=None):
        activations = {}
        def _capture(k, v):
            if k in layer_keys:
                activations[k] = v

        if isinstance(self.encoder, _MultiEncoder):
            if group_idx is None:
                raise ValueError("group_idx must be provided for multi-group encoder")
            x = self.encoder(x, group_idx)
        else:
            x = jax.vmap(self.encoder)(x)
            
        _capture("post_encoder", x)

        for i, block in enumerate(self.ssm_blocks):
            x, state, block_activations = block.call_with_activations(
                x, state, layer_keys=layer_keys
            )
            activations.update({f"{k}_{i}": v for k, v in block_activations.items()})

        x = jax.vmap(self.decoder)(x)
        return x, state, activations
    
    
# class SSMFoundationalDecoder(eqx.Module):
#     encoders: List[eqx.nn.Linear]  # group_key → encoder
#     encoder_dropout: eqx.nn.Dropout
#     ssm_blocks: List[SSMBlock]
#     decoder: eqx.nn.Linear
#     decoder_dropout: eqx.nn.Dropout
#     stateful: bool = True
#     nondeterministic: bool = True
#     lip2: bool = False

#     def __init__(
#         self,
#         rng_seed,
#         input_dim=MAX_NEURAL_UNITS,
#         num_dataset_groups=10,
#         dim_ssm_io=32,  # dim of input and output of the SSM, H in the S5 paper
#         ssm_num_layers=2,  # number of layers of SSMs
#         output_dim=2,  # dim of final output of the model
#         dropout_p: float = 0.3,
#         ssm_dropout_p: float = 0.05,
#         ssm_layer_cls: Union[str, Any] = "S5Layer",
#         ssm_layer_args: Dict = {},
#     ):
#         key = jr.PRNGKey(rng_seed)
#         encoder_key, block_key, decoder_key = jr.split(key, 4)
#         self.encoder_dropout = eqx.nn.Dropout(p=dropout_p)

#         block_keys = jr.split(block_key, ssm_num_layers)

#         # Resolve SSM Layer Class
#         if isinstance(ssm_layer_cls, str):
#             if ssm_layer_cls not in SSM_LAYER_REGISTRY:
#                 raise ValueError(f"Unknown SSM layer class: {ssm_layer_cls}")
#             ssm_layer_cls = SSM_LAYER_REGISTRY[ssm_layer_cls]

#         # Group-specific encoders
#         self.encoders = []
#         enc_keys = jr.split(encoder_key, num_dataset_groups)
#         for gi in range(num_dataset_groups):
#             encoder = eqx.nn.Linear(
#                 input_dim, dim_ssm_io, key=enc_keys[gi]
#             )
#             self.encoders.append(encoder)

#         # Decoder
#         self.decoder = eqx.nn.Linear(
#             dim_ssm_io, output_dim, key=decoder_key
#         )
#         self.decoder_dropout = eqx.nn.Dropout(p=dropout_p)

#         # SSM blocks
#         self.ssm_blocks = [
#             SSMBlock(
#                 dim_ssm_io=dim_ssm_io,
#                 ssm_layer_cls=ssm_layer_cls,
#                 ssm_layer_args=ssm_layer_args,
#                 key=k,
#                 drop_rate=ssm_dropout_p,
#             )
#             for k in block_keys
#         ]

#     def __call__(self, x, state, group_idx, key):
#         """Compute S5 for a specific dataset. Returns output, state, and a dict of intermediate SSM block outputs."""
#         # 1. Project input to SSM dimension
#         encoders_vmap = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
#         x = jax.lax.switch(group_idx, encoders_vmap, x)

#         # Split key for all dropout layers
#         num_dropout_layers = 2 + len(self.ssm_blocks)
#         dropkeys = jr.split(key, num_dropout_layers)

#         x = self.encoder_dropout(x, key=dropkeys[0])

#         # 3. Apply S5 blocks 
#         for i, (block, key) in enumerate(zip(self.ssm_blocks, dropkeys[1:-1])):
#             x, state = block(x, state, key=key)

#         # 4. Project output to behavior dimension
#         x = jax.vmap(self.decoder)(x)
#         x = self.decoder_dropout(x, key=dropkeys[-1])
#         return x, state

#     def call_with_activations(self, x, state, group_idx, layer_keys):
#         activations = {}

#         def _capture(k, v):
#             if k in layer_keys:
#                 activations[k] = v
#                 return

#         """Compute S5 for a specific dataset. Returns output, state, and a dict of intermediate SSM block outputs."""
#         # 1. Project input to SSM dimension
#         encoders_vmap = [jax.vmap(enc, in_axes=0, out_axes=0) for enc in self.encoders]
#         x = jax.lax.switch(group_idx, encoders_vmap, x)
#         _capture("post_encoder", x)

#         # 3. Apply S5 blocks and collect activations
#         key = jr.PRNGKey(0)  # just for compatibility
#         dropkeys = jr.split(key, len(self.ssm_blocks))
#         for i, (block, key) in enumerate(zip(self.ssm_blocks, dropkeys)):
#             x, state, block_activations = block.call_with_activations(
#                 x, state, layer_keys=layer_keys
#             )
#             activations.update({f"{k}_{i}": v for k, v in block_activations.items()})

#         # 4. Project output to behavior dimension
#         x = jax.vmap(self.decoder)(x)
#         return x, state, activations


# class SSMDownstreamDecoder(eqx.Module):
#     encoder: eqx.nn.Linear
#     encoder_dropout: eqx.nn.Dropout
#     ssm_blocks: List[SSMBlock]
#     decoder: eqx.nn.Linear
#     decoder_dropout: eqx.nn.Dropout

#     def __init__(
#         self,
#         rng_seed,
#         input_dim,
#         dim_ssm_state,
#         dim_ssm_io,
#         ssm_num_layers,
#         output_dim,
#         dropout_p: float = 0.1,
#         pretrained_ssm_blocks: Optional[List] = None,
#         C_init: str = "trunc_standard_normal",
#         conj_sym: bool = True,
#         clip_eigs: bool = False,
#         discretisation: str = "zoh",
#         dt_min: float = 0.001,
#         dt_max: float = 0.1,
#     ):
#         key = jr.PRNGKey(rng_seed)
#         encoder_key, block_key, decoder_key = jr.split(key, 3)

#         encoder_in_dim = dim_ssm_io
#         self.encoder = eqx.nn.Linear(input_dim, encoder_in_dim, key=encoder_key)
#         self.encoder_dropout = eqx.nn.Dropout(p=dropout_p)

#         if pretrained_ssm_blocks is not None:
#             self.ssm_blocks = pretrained_ssm_blocks
#         else:
#             block_keys = jr.split(block_key, ssm_num_layers)
#             self.ssm_blocks = [
#                 SSMBlock(
#                     dim_ssm_state=int(dim_ssm_state),
#                     dim_ssm_io=dim_ssm_io,
#                     C_init=C_init,
#                     conj_sym=conj_sym,
#                     clip_eigs=clip_eigs,
#                     discretisation=discretisation,
#                     dt_min=dt_min,
#                     dt_max=dt_max,
#                     drop_rate=dropout_p,
#                     key=key,
#                 )
#                 for key in block_keys
#             ]

#         # Decoder: use pretrained if provided, else initialize new
#         self.decoder = eqx.nn.Linear(dim_ssm_io, output_dim, key=decoder_key)
#         self.decoder_dropout = eqx.nn.Dropout(p=dropout_p)

#     def __call__(self, x, state, key):
#         # Project input to SSM dimension
#         x = jax.vmap(self.encoder)(x)

#         # Split key for all dropout layers
#         num_dropout_layers = 2 + len(self.ssm_blocks)
#         dropkeys = jr.split(key, num_dropout_layers)

#         x = self.encoder_dropout(x, key=dropkeys[0])
#         # x = jax.vmap(self.glu)(x)

#         # Apply SSM blocks
#         for block, k in zip(self.ssm_blocks, dropkeys[1:-1]):
#             x, state = block(x, state, key=k)
#         # Project output to behavior dimension
#         x = jax.vmap(self.decoder)(x)
#         x = self.decoder_dropout(x, key=dropkeys[-1])
#         return x, state

#     def call_with_activations(self, x, state, layer_keys=[]):
#         """
#         Computes S5 and optionally returns a dictionary of intermediate activations.

#         Args:
#             x: Input tensor
#             state: Model state
#             key: PRNG key
#             layer_keys: List of activation keys to capture. Special case: if "ssm_post_activation"
#                         is included, activations for all SSM layers will be captured.
#         """

#         def _capture(k, v):
#             if k in layer_keys:
#                 activations[k] = v
#                 return

#         activations = {}
#         layer_keys = layer_keys

#         # 1. encode + dropout + context
#         x = jax.vmap(self.encoder)(x)
#         _capture("post_encoder", x)

#         # 2. SSM blocks
#         for i, block in enumerate(self.ssm_blocks):
#             x, state, block_activations = block.call_with_activations(
#                 x, state, layer_keys=layer_keys
#             )
#             activations.update({f"{k}_{i}": v for k, v in block_activations.items()})

#         # 3. Project output to behavior dimension
#         x = jax.vmap(self.decoder)(x)
#         return x, state, activations
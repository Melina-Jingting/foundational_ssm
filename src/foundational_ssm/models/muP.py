import jax.numpy as jnp
import jax.random as jr
import math
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
import dataclasses
from collections.abc import Callable
import equinox as eqx


def compute_muP_scale(fan_out, fan_in):
    muP_scale = 1 / jnp.sqrt(fan_in) * jnp.minimum(1.0, jnp.sqrt(fan_out / fan_in))
    return muP_scale

def muP_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    scale = compute_muP_scale(shape[0], shape[1])
    return jr.normal(key, shape, dtype) * scale


def make_muP_init(fan_out_override: int | None = None, fan_in_override: int | None = None):
    """
    Factory for a muP initializer compatible with Linear.init_fn signature.

    - If overrides are provided, they will be used instead of inferring from shape.
    - Returns a function (key, shape, dtype, lim) -> jax.Array
    """

    def _init(key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float) -> jax.Array:
        # Infer fan-out/in from weight shapes when not overridden.
        # For matrices, shape = (fan_out, fan_in). For vectors/bias, fall back to fan_in=1.
        if len(shape) == 2:
            fo = fan_out_override if fan_out_override is not None else shape[0]
            fi = fan_in_override if fan_in_override is not None else shape[1]
        elif len(shape) == 1:
            fo = fan_out_override if fan_out_override is not None else shape[0]
            fi = fan_in_override if fan_in_override is not None else 1
        else:
            # Generic fallback: treat last dim as fan_in, first as fan_out.
            fo = fan_out_override if fan_out_override is not None else shape[0]
            fi = fan_in_override if fan_in_override is not None else shape[-1]

        scale = compute_muP_scale(fo, fi)
        return jr.normal(key, shape, dtype) * scale

    return _init
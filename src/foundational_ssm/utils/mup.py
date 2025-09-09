from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Callable 
from functools import partial

import equinox as eqx
import jax
import jax.tree as jt
import jax.numpy as jnp
import optax
import chex

# ----------------------------- metadata ------------------------------------


@dataclass(frozen=True)
class MupMeta:
    """Per-parameter muP metadata.

    dims: tuple with length equal to the array rank; for each axis:
      - None -> axis size unchanged between base and target
      - float ratio -> target_axis_size / base_axis_size for axes that scale with width
    """

    dims: Tuple[Optional[float], ...]

    @property
    def ndims(self) -> int:
        return sum(1 for d in self.dims if d is not None)

    @property
    def width(self) -> float:
        # Use the last non-None axis ratio as the width of the final transformation
        for d in reversed(self.dims):
            if d is not None:
                return float(d)
        return 1.0

    def _changed_axes(self) -> Tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.dims) if d is not None)

    def is_hidden_weight(self) -> bool:
        return self.ndims == 2

    def is_vector_like(self) -> bool:
        return self.ndims == 1

    def classify_vector_like(self, axis_convention: str) -> Tuple[bool, bool]:
        """Return (is_input_weight, is_output_weight) for vector-like params.

        axis_convention:
          - "flax": input weight if the changed axis is the last axis.
          - "torch": input weight if the changed axis is the first axis.
        """
        if not self.is_vector_like():
            return (False, False)
        changed = self._changed_axes()[0]
        last_axis = len(self.dims) - 1
        if axis_convention == "flax":
            is_input = changed == last_axis
        else:  # "torch" default for Equinox/PyTorch-like shapes
            is_input = changed == 0
        is_output = not is_input
        return (is_input, is_output)


def _leaf_shape(x: Any) -> Optional[Tuple[int, ...]]:
    return tuple(x.shape) if isinstance(x, jnp.ndarray) else None

def to_shape(tree):
    return jt.map(_leaf_shape, tree)

def build_mup_meta(base_tree: Any, target_tree: Any) -> Any:
    """Construct a PyTree of MupMeta aligned with `target_tree`'s leaves.

    base_tree and target_tree must have identical PyTree structures on array leaves.
    """

    base_shapes = to_shape(base_tree)
    tgt_shapes = to_shape(target_tree)
    base_shapes_leaves, base_treedef = jax.tree_util.tree_flatten(base_shapes, is_leaf=lambda x: isinstance(x, tuple))
    tgt_shapes_leaves, target_treedef = jax.tree_util.tree_flatten(tgt_shapes, is_leaf=lambda x: isinstance(x, tuple))

    mup_shapes = []
    for bshape,tshape in zip(base_shapes_leaves, tgt_shapes_leaves):
        dims = tuple((None if bi == ti else float(ti) / float(bi)) for bi, ti in zip(bshape, tshape))
        mup_shapes.append(MupMeta(dims))
    meta = jt.unflatten(target_treedef, mup_shapes)

    return meta


def apply_mup_init_rescale(model: Any, meta_tree: Any, axis_convention: str = "torch") -> Any:
    """Rescale initialized parameters for muP (output weights scaled by 1/sqrt(width)).

    Returns a new model with updated parameters.
    """

    def maybe_rescale(param, meta: Optional[MupMeta]):
        if not isinstance(param, jnp.ndarray) or not isinstance(meta, MupMeta):
            return param
        if meta.is_hidden_weight():  # hidden weights are not rescaled at init (only updates)
            return param
        is_input, is_output = meta.classify_vector_like(axis_convention)
        if is_output:
            return param / jnp.sqrt(jnp.asarray(meta.width, dtype=param.dtype))
        return param

    return jt.map(maybe_rescale, eqx.filter(model, eqx.is_array), meta_tree)


# ------------------------- optax transformations ----------------------------

tree_map_mupped = partial(
    jax.tree_util.tree_map,
    is_leaf=lambda leaf: isinstance(leaf, MupMeta),
)

def scale_adam_by_mup(meta_tree: Any, axis_convention: str = "torch") -> optax.GradientTransformation:
    """Scale Adam-like updates according to μP using a parallel meta tree."""

    def _scale(update, meta):
        # Only act when there's metadata + an array update
        if not isinstance(meta, MupMeta) or not isinstance(update, jnp.ndarray):
            return update

        # Hidden weights: scale by 1/width
        if meta.is_hidden_weight():
            return update / jnp.asarray(meta.width, dtype=update.dtype)

        # Vector-like: scale output weights by 1/width
        is_input, is_output = meta.classify_vector_like(axis_convention)
        if is_output:
            return update / jnp.asarray(meta.width, dtype=update.dtype)

        return update

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        scaled = jax.tree_util.tree_map(
            _scale, updates, meta_tree,
            is_leaf=lambda x: isinstance(x, MupMeta)
        )
        return scaled, state

    return optax.GradientTransformation(init_fn, update_fn)


# ------------------------------- helpers ------------------------------------


def infer_and_apply_mup(
    base_model: Any,
    target_model: Any,
    *,
    axis_convention: str = "torch",
) -> tuple[Any, Any]:
    """Convenience helper: build meta from base/target arrays and rescale target init.

    Returns (meta_tree, target_model_rescaled).
    Keep `meta_tree` for the optimizer scaling throughout training.
    """

    base_params = eqx.filter(base_model, eqx.is_array)
    target_params = eqx.filter(target_model, eqx.is_array)
    meta = build_mup_meta(base_params, target_params)
    target_rescaled = apply_mup_init_rescale(target_model, meta, axis_convention=axis_convention)
    return meta, target_rescaled

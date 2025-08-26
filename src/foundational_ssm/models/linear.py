import math
from typing import Any, Literal, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray
import dataclasses
from collections.abc import Callable
import equinox as eqx


def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    if jnp.issubdtype(dtype, jnp.complexfloating):
        real_dtype = jnp.finfo(dtype).dtype
        rkey, ikey = jrandom.split(key, 2)
        real = jrandom.uniform(rkey, shape, real_dtype, minval=-lim, maxval=lim)
        imag = jrandom.uniform(ikey, shape, real_dtype, minval=-lim, maxval=lim)
        return real.astype(dtype) + 1j * imag.astype(dtype)
    else:
        return jrandom.uniform(key, shape, dtype, minval=-lim, maxval=lim)



def default_floating_dtype():
    if jax.config.jax_enable_x64:  # pyright: ignore
        return jnp.float64
    else:
        return jnp.float32

def field(
    *,
    converter: Callable[[Any], Any] | None = None,
    static: bool = False,
    **kwargs: Any,
) -> Any:
    """Equinox supports extra functionality on top of the default dataclasses.

    **Arguments:**

    - `converter`: a function to call on this field when the model is initialised. For
        example, `field(converter=jax.numpy.asarray)` to convert
        `bool`/`int`/`float`/`complex` values to JAX arrays. This is ran after the
        `__init__` method (i.e. when using a user-provided `__init__`), and after
        `__post_init__` (i.e. when using the default dataclass initialisation).
        If `converter` is `None`, then no converter is registered.
    - `static`: whether the field should not interact with any JAX transform at all (by
        making it part of the PyTree structure rather than a leaf).
    - `**kwargs`: All other keyword arguments are passed on to `dataclass.field`.

    !!! example "Example for `converter`"

        ```python
        class MyModule(eqx.Module):
            foo: Array = eqx.field(converter=jax.numpy.asarray)

        mymodule = MyModule(1.0)
        assert isinstance(mymodule.foo, jax.Array)
        ```

    !!! example "Example for `static`"

        ```python
        class MyModule(eqx.Module):
            normal_field: int
            static_field: int = eqx.field(static=True)

        mymodule = MyModule("normal", "static")
        leaves, treedef = jax.tree_util.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```

    `static=True` means that this field is not a node of the PyTree, so it does not
    interact with any JAX transforms, like JIT or grad. This means that it is usually a
    bug to make JAX arrays be static fields. `static=True` should very rarely be used.
    It is preferred to just filter out each field with `eqx.partition` whenever you need
    to select only some fields.
    """
    try:
        metadata = dict(kwargs.pop("metadata"))  # safety copy
    except KeyError:
        metadata = {}
    if "converter" in metadata:
        raise ValueError("Cannot use metadata with `converter` already set.")
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    # We don't just use `lambda x: x` as the default, so that this works:
    # ```
    # class Abstract(eqx.Module):
    #     x: int = eqx.field()
    #
    # class Concrete(Abstract):
    #    @property
    #    def x(self):
    #        pass
    # ```
    # otherwise we try to call the default converter on a property without a setter,
    # and an error is raised.
    # Oddities like the above are to be discouraged, of course, but in particular
    # `field(init=False)` was sometimes used to denote an abstract field (prior to the
    # introduction of `AbstractVar`), so we do want to support this.
    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return dataclasses.field(metadata=metadata, **kwargs)


class Linear(eqx.Module):
    """Performs a linear transformation."""

    weight: Array
    bias: Array | None
    in_features: int | Literal["scalar"] = field(static=True)
    out_features: int | Literal["scalar"] = field(static=True)
    use_bias: bool = field(static=True)


    def __init__(
        self,
        in_features: int | Literal["scalar"],
        out_features: int | Literal["scalar"],
        use_bias: bool = True,
        dtype=None,
        init_fn: Callable = default_init,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        if in_features_ == 0:
            lim = 1.0
        else:
            lim = 1 / math.sqrt(in_features_)
        wshape = (out_features_, in_features_)
        self.weight = init_fn(wkey, wshape, dtype, lim)
        bshape = (out_features_,)
        self.bias = init_fn(bkey, bshape, dtype, lim) if use_bias else None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x

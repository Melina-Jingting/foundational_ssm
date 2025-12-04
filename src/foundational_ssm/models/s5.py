"""
S5 implementation modified from: https://github.com/lindermanlab/S5/blob/main/s5/ssm_init.py

This module implements S5 using JAX and Equinox.

Attributes of the S5 model:
- `linear_encoder`: The linear encoder applied to the input time series.
- `blocks`: A list of S5 blocks, each consisting of an S5 layer, normalisation, GLU, and dropout.
- `linear_layer`: The final linear layer that outputs the predictions of the model.
- `classification`: A boolean indicating whether the model is used for classification tasks.
- `output_step`: For regression tasks, specifies how many steps to skip before outputting a prediction.

The module also includes:
- `S5Layer`: Implements the core S5 layer using structured state space models with options for
  different discretisation methods and eigenvalue clipping.
- `S5Block`: Combines the S5 layer with batch normalisation, a GLU activation, and dropout.
- Utility functions for initialising and discretising the state space model components,
  such as `make_HiPPO`, `make_NPLR_HiPPO`, and `make_DPLR_HiPPO`.
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import lecun_normal, normal
from jax.scipy.linalg import block_diag

from foundational_ssm.models.discretisations import discretise
from foundational_ssm.models.ssm_scan import compute_hidden_states
from foundational_ssm.models.initialisations import init_log_steps


def make_HiPPO(N):
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix
    """
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size

    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = jnp.sqrt(jnp.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:

    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, jnp.newaxis] * P[jnp.newaxis, :]

    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig



def init_VinvB(init_fun, rng, shape, Vinv):
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax jr key to be used with init function.
         shape (tuple): desired shape  (dim_ssm_state, dim_ssm_io)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (dim_ssm_state, dim_ssm_io, 2)
    """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return jnp.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """Sample C with a truncated normal distribution with standard deviation 1.
    Args:
        key: jax jr key
        shape (tuple): desired shape, of length 3, (dim_ssm_io, dim_ssm_state, _)
    Returns:
        sampled C matrix (float32) of shape (dim_ssm_io, dim_ssm_state, 2) (for complex parameterization)
    """
    dim_ssm_io, dim_ssm_state, _ = shape
    Cs = []
    for i in range(dim_ssm_io):
        key, skey = jr.split(key)
        C = lecun_normal()(skey, shape=(1, dim_ssm_state, 2))
        Cs.append(C)
    return jnp.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """Initialize C_tilde=CV. First sample C. Then compute CV.
    Note we will parameterize this with two different matrices for complex
    numbers.
     Args:
         init_fun:  the initialization function to use, e.g. lecun_normal()
         rng:       jax jr key to be used with init function.
         shape (tuple): desired shape  (dim_ssm_io, dim_ssm_state)
         V: (complex64)     the eigenvectors used for initialization
     Returns:
         C_tilde (complex64) of shape (dim_ssm_io, dim_ssm_state, 2)
    """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return jnp.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)



class S5Layer(eqx.Module):
    Lambda_re: jax.Array
    Lambda_im: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    log_step: jax.Array

    dim_ssm_io: int
    dim_ssm_state: int
    conj_sym: bool
    clip_eigs: bool = False
    discretisation: str = "zoh"
    step_rescale: float = 1.0

    def __init__(
        self,
        *,
        dim_ssm_state,
        dim_ssm_io,
        dt_min,
        dt_max,
        discretisation = "zoh",
        C_init="trunc_standard_normal",
        conj_sym = True,
        clip_eigs = False,
        step_rescale=1.0,
        blocks = 4,
        a_initialisation = "s5", # for API consistency
        rand_real = False, # for API consistency
        rand_imag = False, # for API consistency
        key,
    ):
        B_key, C_key, D_key, step_key, key = jr.split(key, 5)

        block_size = int(dim_ssm_state / blocks)
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)

        if conj_sym:
            block_size = block_size // 2
            effective_dim_ssm_state = dim_ssm_state // 2
        else:
            effective_dim_ssm_state = dim_ssm_state

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

        self.dim_ssm_io = dim_ssm_io
        self.dim_ssm_state = effective_dim_ssm_state
        local_dim_ssm_state = 2 * effective_dim_ssm_state

        self.Lambda_re = Lambda.real
        self.Lambda_im = Lambda.imag

        self.conj_sym = conj_sym

        self.clip_eigs = clip_eigs

        self.B = init_VinvB(lecun_normal(), B_key, (local_dim_ssm_state, self.dim_ssm_io), Vinv)

        # Initialize state to output (C) matrix
        if C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
        elif C_init in ["lecun_normal"]:
            C_init = lecun_normal()
        elif C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError("C_init method {} not implemented".format(C_init))
        
        if C_init in ["complex_normal"]:
            self.C = C_init(C_key, (self.dim_ssm_io, 2 * self.dim_ssm_state, 2))
        else:
            self.C = init_CV(C_init, C_key, (self.dim_ssm_io, local_dim_ssm_state, 2), V)

        self.D = normal(stddev=1.0)(D_key, (self.dim_ssm_io,))

        # Initialize learnable discretisation timescale value
        self.log_step = init_log_steps(step_key, self.dim_ssm_state, dt_min, dt_max)

        self.step_rescale = step_rescale
        self.discretisation = discretisation

    def __call__(self, input_sequence):
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        step = self.step_rescale * jnp.exp(self.log_step)
        Lambda_bar, B_bar = discretise(self.discretisation, Lambda, B_tilde, step)
        
        xs = compute_hidden_states(Lambda_bar, B_bar, input_sequence)
        ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du

    def call_with_activations(self, input_sequence):
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        step = self.step_rescale * jnp.exp(self.log_step)

        # Discretize
        Lambda_bar, B_bar = discretise(self.discretisation, Lambda, B_tilde, step)

        xs = compute_hidden_states(Lambda_bar, B_bar, input_sequence)
        ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du, xs

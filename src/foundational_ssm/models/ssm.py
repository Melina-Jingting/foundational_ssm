import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn.initializers import lecun_normal, normal
from foundational_ssm.models.initialisations import init_lambda, init_log_steps
from foundational_ssm.models.discretisations import discretise
from foundational_ssm.models.ssm_scan import compute_hidden_states

class ContinuousSSMLayer(eqx.Module):
    Lambda_re: jax.Array
    Lambda_im: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    log_step: jax.Array

    dim_ssm_io: int
    dim_ssm_state: int
    conj_sym: bool = True
    clip_eigs: bool = False
    discretisation: str = "zoh"
    a_initialisation: str = "s4d_inv"
    step_rescale: float = 1.0

    def __init__(
        self,
        dim_ssm_state,
        dim_ssm_io,
        a_initialisation,
        discretisation,
        dt_min,
        dt_max,
        *,
        key,
    ):
        A_key, B_key, C_key, D_key, step_key, key = jr.split(key, 6)

        self.dim_ssm_io = dim_ssm_io
        self.dim_ssm_state = dim_ssm_state
    
        Lambda = init_lambda(a_initialisation, dim_ssm_state // 2)
        self.Lambda_re = Lambda.real
        self.Lambda_im = Lambda.imag
        self.B = lecun_normal()(B_key, (dim_ssm_state, self.dim_ssm_io))
        self.C = lecun_normal(batch_axis=0)(C_key, (self.dim_ssm_io, 2 * self.dim_ssm_state, 2))
        self.D = normal(stddev=1.0)(D_key, (self.dim_ssm_io,))

        # Initialize learnable discretisation timescale value
        self.log_step = init_log_steps(step_key, self.dim_ssm_state, dt_min, dt_max)

        self.discretisation = discretisation

    def __call__(self, input_sequence):
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        step = self.step_rescale * jnp.exp(self.log_step[:, 0])
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

        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretize
        Lambda_bar, B_bar = discretise(self.discretisation, Lambda, B_tilde, step)

        xs = compute_hidden_states(Lambda_bar, B_bar, input_sequence)
        ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du, xs
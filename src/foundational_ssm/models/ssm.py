import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Optional
from jax.nn.initializers import lecun_normal, normal
from foundational_ssm.models.initialisations import init_lambda, init_log_steps
from foundational_ssm.models.discretisations import discretise
from foundational_ssm.models.ssm_scan import compute_hidden_states

class ContinuousSSMLayer(eqx.Module):
    Lambda_re: jax.Array # (P/2,) or (P,)
    Lambda_im: Optional[jax.Array]
    B: jax.Array # (P/2, H, 2) or (P, H)
    C: jax.Array # (H, P/2, 2) or (H, P)
    D: jax.Array # (H,)
    log_step: jax.Array # (P/2,) or (P,)

    dim_ssm_io: int
    dim_ssm_state: int
    conj_sym: bool = True
    discretisation: str = "zoh"
    a_initialisation: str = "s4d_inv"
    step_rescale: float = 1.0

    def __init__(
        self,
        *,
        dim_ssm_state,
        dim_ssm_io,
        dt_min,
        dt_max,
        a_initialisation = "s4d_inv",
        discretisation = "zoh",
        conj_sym = True,
        rand_real = False,
        rand_imag = False,
        key,
    ):
        A_key, B_key, C_key, D_key, step_key, key = jr.split(key, 6)

        self.dim_ssm_io = dim_ssm_io
        self.dim_ssm_state = dim_ssm_state
        self.conj_sym = conj_sym

        if conj_sym:
            effective_dim_state = dim_ssm_state // 2
        else:
            effective_dim_state = dim_ssm_state
    
        Lambda = init_lambda(a_initialisation, effective_dim_state, rand_real, rand_imag, rand_key=A_key)
        self.Lambda_re = Lambda.real
        
        if conj_sym:
            self.Lambda_im = Lambda.imag
            self.B = lecun_normal(batch_axis=0)(B_key, (effective_dim_state, self.dim_ssm_io, 2)) / jnp.sqrt(2.0)
            self.C = lecun_normal(batch_axis=0)(C_key, (self.dim_ssm_io, effective_dim_state, 2)) / jnp.sqrt(2.0)
        else:
            self.Lambda_im = None
            self.B = lecun_normal(batch_axis=0)(B_key, (effective_dim_state, self.dim_ssm_io))
            self.C = lecun_normal(batch_axis=0)(C_key, (self.dim_ssm_io, effective_dim_state))

        self.D = normal(stddev=1.0)(D_key, (self.dim_ssm_io,))

        self.log_step = init_log_steps(step_key, effective_dim_state, dt_min, dt_max)

        self.discretisation = discretisation

    def __call__(self, input_sequence):
        if self.conj_sym:
            Lambda = self.Lambda_re + 1j * self.Lambda_im
            B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
            C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
        else:
            Lambda = self.Lambda_re
            B_tilde = self.B
            C_tilde = self.C

        step = self.step_rescale * jnp.exp(self.log_step)
        Lambda_bar, B_bar = discretise(self.discretisation, Lambda, B_tilde, step)
        
        xs = compute_hidden_states(Lambda_bar, B_bar, input_sequence)
        
        if self.conj_sym:
            ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
        else:
            ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs)
            
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du

    def call_with_activations(self, input_sequence):
        if self.conj_sym:
            Lambda = self.Lambda_re + 1j * self.Lambda_im
            B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
            C_tilde = self.C[..., 0] + 1j * self.C[..., 1]
        else:
            Lambda = self.Lambda_re
            B_tilde = self.B
            C_tilde = self.C

        step = self.step_rescale * jnp.exp(self.log_step)

        # Discretize
        Lambda_bar, B_bar = discretise(self.discretisation, Lambda, B_tilde, step)

        xs = compute_hidden_states(Lambda_bar, B_bar, input_sequence)
        
        if self.conj_sym:
            ys = jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
        else:
            ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du, xs
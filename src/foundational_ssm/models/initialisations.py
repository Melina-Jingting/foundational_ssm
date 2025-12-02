import jax.numpy as jnp 
import numpy as np
import jax.random as jr 

def init_s4d_inv(N):
    return -0.5 + 1j * N/np.pi * (N / (2* jnp.arange(N) + 1) - 1)

def init_s4d_lin(N):
    return -0.5 + 1j * np.pi * np.arange(N//2)

def init_s4d_real(N):
    return -( jnp.arange(N) + 1 )
    

def init_lambda(method: str, N, rand_real=False, rand_imag=False, rand_key=None):
    if method == "s4d_inv":
        Lambda = init_s4d_inv(N)
    elif method == "s4d_lin":
        Lambda = init_s4d_lin(N)
    elif method == "s4d_real":
        Lambda = init_s4d_real(N)
    else:
        raise NotImplementedError(f"Unknown initialization method: {method}")
    
    # For randomisation
    if (rand_real or rand_imag) and rand_key is None:
        raise ValueError("rand_key must be provided if rand_real or rand_imag is True")
    if rand_real:
        real_key, rand_key = jr.split(rand_key)
        Lambda = Lambda.at[:].set(Lambda.real + jr.normal(real_key, Lambda.real.shape) + Lambda.imag * 1j)
    if rand_imag:
        imag_key, rand_key = jr.split(rand_key)
        Lambda = Lambda.at[:].set(Lambda.real + (jr.normal(imag_key, Lambda.imag.shape)) * 1j)
    
    return Lambda    
    

def init_log_steps(key, N, dt_min, dt_max):
    """Initialize an array of learnable timescale parameters
    Args:
        key: jax jr key
        input: tuple containing the array shape H and
               dt_min and dt_max
    Returns:
        initialized array of timescales (float32): (H,)
    """
    return jr.uniform(key, (N,)) * (jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
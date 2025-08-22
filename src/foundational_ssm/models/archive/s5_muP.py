class S5Layer(eqx.Module):
    Lambda_re: jax.Array
    Lambda_im: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    log_step: jax.Array

    H: int
    P: int
    conj_sym: bool
    clip_eigs: bool = False
    discretisation: str = "zoh"
    step_rescale: float = 1.0

    def __init__(
        self,
        ssm_size,
        blocks,
        H,
        C_init, # This argument will be overridden by muP-SSM
        conj_sym,
        clip_eigs,
        discretisation,
        dt_min,
        dt_max,
        step_rescale,
        *,
        key
    ):

        B_key, C_key, D_key, step_key, key = jr.split(key, 5)

        block_size = int(ssm_size / blocks)
        # Initialize state matrix A using approximation to HiPPO-LegS matrix
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        if conj_sym:
            block_size = block_size // 2
            P = ssm_size // 2
        else:
            P = ssm_size

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block
        Lambda = (Lambda * jnp.ones((blocks, block_size))).ravel()
        V = block_diag(*([V] * blocks))
        Vinv = block_diag(*([Vc] * blocks))

        self.H = H
        self.P = P
        if conj_sym:
            local_P = 2 * P
        else:
            local_P = P

        self.Lambda_re = Lambda.real
        self.Lambda_im = Lambda.imag

        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs

        # --- muP-SSM Parameterization Start ---
        # Let Nx = local_P (state size) and Nu = self.H (output dim)
        # According to muP-SSM theory, the standard deviations for B and C
        # should be scaled for stable feature learning.[1]

        # 1. Initialization scaling for B
        stddev_B = jnp.sqrt(local_P / self.H)
        B_init_fn = normal(stddev=stddev_B)
        self.B = init_VinvB(B_init_fn, B_key, (local_P, self.H), Vinv)

        # 2. Initialization scaling for C
        # This principled scaling replaces the previous heuristic C_init options.
        stddev_C = 1.0 / jnp.sqrt(local_P * self.H)
        C_init_fn = normal(stddev=stddev_C)
        self.C = init_CV(C_init_fn, C_key, (self.H, local_P, 2), V)
        # --- muP-SSM Parameterization End ---

        self.D = normal(stddev=1.0)(D_key, (self.H,))

        # Initialize learnable discretisation timescale value
        self.log_step = init_log_steps(step_key, (self.P, dt_min, dt_max))

        self.step_rescale = step_rescale
        self.discretisation = discretisation

    def __call__(self, input_sequence):
        if self.clip_eigs:
            Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretize
        if self.discretisation in ["zoh"]:
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretisation in ["bilinear"]:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError(
                "Discretization method {} not implemented".format(self.discretisation)
            )

        ys = apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, self.conj_sym)

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du
    
    
import optax

# Assume 'model' is an instance of a larger network containing your S5Layer(s)
# and 'base_lr' is your chosen base learning rate (e.g., 1e-3).

# 1. Define a function to partition the model parameters.
# This uses Equinox's filtering capabilities to identify the different matrices.
def partition_fn(model):
    # Get the PyTree definition of the model
    flat_model, treedef = jax.tree_util.tree_flatten_with_path(model)
    
    # Identify paths to different parameters
    # Note: This depends on how S5Layer is nested in your full model.
    # You may need to adjust the path checks.
    is_A = lambda path: 'Lambda_re' in str(path) or 'Lambda_im' in str(path)
    is_B = lambda path: 'B' in str(path)
    is_C = lambda path: 'C' in str(path)
    
    # Create a PyTree of labels for each parameter
    partition = {}
    for key, leaf in flat_model:
        path_str = jax.tree_util.keystr(key)
        if is_A(path_str):
            label = "A"
        elif is_B(path_str):
            label = "B"
        elif is_C(path_str):
            label = "C"
        else:
            label = "other" # For D, log_step, norms, etc.
        
        # Assign the label to the parameter's position in the PyTree
        # This is a bit complex; a simpler way is to use eqx.partition
        # For demonstration, we'll use a conceptual approach.
    
    # A much simpler way with eqx.partition:
    is_A_fn = lambda m: (m.Lambda_re, m.Lambda_im)
    is_B_fn = lambda m: m.B
    is_C_fn = lambda m: m.C
    
    # This assumes you can filter directly on an S5Layer instance
    # In a full model, you'd recursively apply this.
    s5_layer = model # Or model.path.to.s5_layer
    
    return eqx.partition(
        s5_layer,
        (is_A_fn, is_B_fn, is_C_fn),
        is_leaf=lambda x: isinstance(x, S5Layer)
    )


# 2. Define the learning rate multipliers based on muP-SSM rules.
# Assume s5_layer is an instance of your S5Layer to get H and P.
H = s5_layer.H
P = s5_layer.P # Or local_P if you need to account for conj_sym
if s5_layer.conj_sym:
    local_P = 2 * P
else:
    local_P = P

# muP-SSM learning rate scalings [1]
lr_A_mult = H
lr_B_mult = jnp.sqrt(local_P / H)
lr_C_mult = 1.0 / (local_P * jnp.sqrt(H))

# 3. Create a multi-transform optimizer in Optax.
# This example uses Adam, but it works with any optimizer.
optimizer = optax.multi_transform(
    {
        "A": optax.adam(learning_rate=base_lr * lr_A_mult),
        "B": optax.adam(learning_rate=base_lr * lr_B_mult),
        "C": optax.adam(learning_rate=base_lr * lr_C_mult),
        "other": optax.adam(learning_rate=base_lr), # Use base LR for other params
    },
    # This mapping function tells the optimizer which group a parameter belongs to.
    # You would need a robust way to label your parameters.
    param_labels=partition_fn(model) 
)
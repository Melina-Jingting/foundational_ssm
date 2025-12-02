import jax
import jax.numpy as jnp


@jax.vmap
def linear_recurrence_op(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def compute_hidden_states(Lambda_bar, B_bar, input_sequence):
    """Compute the LxH output of discretized SSM given an LxH input.
    Args:
        Lambda_bar (complex64): discretized diagonal state matrix    (P,)
        B_bar      (complex64): discretized input matrix             (P, H)
        C_tilde    (complex64): output matrix                        (H, P)
        input_sequence (float32): input sequence of features         (L, H)
        conj_sym (bool):         whether conjugate symmetry is enforced
    Returns:
        ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * jnp.ones(
        (input_sequence.shape[0], Lambda_bar.shape[0])
    )
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(linear_recurrence_op, (Lambda_elements, Bu_elements))
    return xs


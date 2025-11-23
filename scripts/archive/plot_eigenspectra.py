import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import os
from typing import Dict, List, Tuple, Optional
import equinox as eqx

# Import model classes
from foundational_ssm.models import S5, LRU, LinOSS
from foundational_ssm.models.s4_equinox import S4Layer, make_HiPPO, make_DPLR_HiPPO
from foundational_ssm.models.s4d import S4DKernel

# Create output directory
os.makedirs("eigenspectra_plots", exist_ok=True)

# Set random seeds for reproducibility
jax_key = random.PRNGKey(42)
torch.manual_seed(42)
np.random.seed(42)

# Model configuration
config = {
    "hidden_dim": 64,
    "state_dim": 64,
    "timestep": 0.005,  # For discrete-time conversion
    "num_blocks": 1,  # Visualize just one block
    "input_dim": 1,  # Doesn't matter for eigenvalue analysis
    "output_dim": 1,  # Doesn't matter for eigenvalue analysis
    "sequence_length": 1024,  # Required for S4
}


def visualize_discrete_eigenvalues(
    lambdas_disc_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Plot discrete time eigenvalues for multiple models

    Args:
        lambdas_disc_dict: Dictionary mapping model names to eigenvalues
        save_path: Path to save the figure, if None, just displays
        title: Optional custom title
    """
    # Create figure with light theme for better visibility on light backgrounds
    plt.style.use("default")

    # Create a figure with subplots in a grid
    n_models = len(lambdas_disc_dict)
    n_cols = 3  # 3 columns layout
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division for number of rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    fig.patch.set_alpha(0.0)  # Transparent background

    # Color palette for different models - using darker colors for light background
    colors = {
        "S4": "#0077B6",  # Deep blue
        "S4D": "#D62828",  # Deep red
        "S5": "#006400",  # Dark green
        "LRU": "#6A0DAD",  # Purple
        "LinOSS-IMEX": "#B26B00",  # Brown
        "LinOSS-IM": "#7B3F00",  # Dark brown
    }

    # Flatten axes if multiple rows
    if n_rows > 1:
        axes = axes.flatten()

    # Create individual plot for each model
    for i, (model_name, lambdas) in enumerate(lambdas_disc_dict.items()):
        # Get current axis
        if n_models == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[i]
        else:
            ax = axes[i]

        # Plot eigenvalues
        ax.scatter(
            lambdas.real,
            lambdas.imag,
            s=50,
            color=colors[model_name],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add grid and axes
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_title(f"{model_name}", fontsize=16)
        ax.set_aspect("equal")

        # Add unit circle to show stability region
        unit_circle = Circle(
            (0, 0),
            1,
            fill=False,
            color="#D62828",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        ax.add_patch(unit_circle)

        # Set consistent axes limits for all plots
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        # Add ticks and labels
        ax.tick_params(axis="both", colors="black")

        # Create and save individual plot for each model
        fig_individual, ax_individual = plt.subplots(figsize=(5, 5))
        fig_individual.patch.set_alpha(0.0)  # Transparent background

        # Plot eigenvalues in individual figure
        ax_individual.scatter(
            lambdas.real,
            lambdas.imag,
            s=60,
            color=colors[model_name],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

        # Add grid and axes to individual plot
        ax_individual.axhline(
            y=0, color="black", linestyle="--", alpha=0.3, linewidth=1
        )
        ax_individual.axvline(
            x=0, color="black", linestyle="--", alpha=0.3, linewidth=1
        )
        ax_individual.grid(True, alpha=0.2, linestyle=":")
        ax_individual.set_title(f"{model_name} Discrete Eigenvalues", fontsize=18)
        ax_individual.set_xlabel("Real Part", fontsize=14)
        ax_individual.set_ylabel("Imaginary Part", fontsize=14)
        ax_individual.set_aspect("equal")

        # Add unit circle to individual plot
        unit_circle_individual = Circle(
            (0, 0),
            1,
            fill=False,
            color="#D62828",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        ax_individual.add_patch(unit_circle_individual)

        # Set consistent axes limits for individual plot
        ax_individual.set_xlim(-1.1, 1.1)
        ax_individual.set_ylim(-1.1, 1.1)

        plt.tight_layout()

        # Save individual plot
        if save_path:
            individual_path = save_path.replace(
                ".png", f"_{model_name.lower().replace('-', '_')}.png"
            )
            plt.savefig(
                individual_path, dpi=300, bbox_inches="tight", transparent=False
            )
            print(f"Saved {model_name} eigenvalue plot to {individual_path}")
        else:
            plt.show()

        plt.close(fig_individual)

    # Hide any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        if n_rows > 1:
            axes[j].axis("off")
        elif n_rows == 1 and n_models < n_cols:
            axes[j].axis("off")

    # Add common labels and title
    fig.text(0.5, 0.02, "Real Part", ha="center", fontsize=14)
    fig.text(0.02, 0.5, "Imaginary Part", va="center", rotation="vertical", fontsize=14)

    if title:
        fig.suptitle(title, fontsize=20)
    else:
        fig.suptitle(
            f"Discrete-time Eigenvalues (dt={config['timestep']})", fontsize=20
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)
        print(f"Saved grid eigenvalue plot to {save_path}")
    else:
        plt.show()

    plt.close()

    # Also create a combined plot with all models in one figure
    fig_combined, ax_combined = plt.subplots(figsize=(12, 10))
    fig_combined.patch.set_alpha(1.0)  # White background

    # Plot each model's eigenvalues on a single plot
    for model_name, lambdas in lambdas_disc_dict.items():
        ax_combined.scatter(
            lambdas.real,
            lambdas.imag,
            s=50,
            color=colors[model_name],
            alpha=0.8,
            label=model_name,
            edgecolors="black",
            linewidths=0.5,
        )

    # Add grid and axes
    ax_combined.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax_combined.axvline(x=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax_combined.grid(True, alpha=0.2, linestyle=":")
    ax_combined.set_title(
        f"Combined Discrete-time Eigenvalues (dt={config['timestep']})", fontsize=18
    )
    ax_combined.set_xlabel("Real Part", fontsize=14)
    ax_combined.set_ylabel("Imaginary Part", fontsize=14)
    ax_combined.set_aspect("equal")

    # Add unit circle to show stability region
    unit_circle = Circle(
        (0, 0), 1, fill=False, color="#D62828", linestyle="--", linewidth=2, alpha=0.7
    )
    ax_combined.add_patch(unit_circle)

    # Add legend with custom styling
    legend = ax_combined.legend(loc="upper right", frameon=True, fontsize=12)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.8)
    frame.set_edgecolor("black")

    # Adjust axes limits for better visualization
    ax_combined.set_xlim(-1.1, 1.1)
    ax_combined.set_ylim(-1.1, 1.1)

    plt.tight_layout()

    # Save combined plot
    if save_path:
        combined_path = save_path.replace(".png", "_combined.png")
        plt.savefig(combined_path, dpi=300, bbox_inches="tight", transparent=False)
        print(f"Saved combined eigenvalue plot to {combined_path}")
    else:
        plt.show()

    plt.close()


# 1. Initialize S4 Model (JAX/Equinox)
def extract_s4_eigenvalues():
    N = config["state_dim"]
    dt = config["timestep"]

    # Initialize S4 layer directly to access the eigenvalues
    s4_layer = S4Layer(N=N, l_max=config["sequence_length"], key=jax_key)

    # Extract continuous-time eigenvalues
    Lambda_re = s4_layer.Lambda_re
    Lambda_im = s4_layer.Lambda_im

    # Continuous-time eigenvalues
    Lambda = jnp.clip(Lambda_re, None, -1e-4) + 1j * Lambda_im

    # Convert to numpy
    Lambda = np.array(Lambda)

    # Discrete-time eigenvalues
    Lambda_disc = np.exp(Lambda * dt)

    return Lambda_disc


# 2. Initialize S4D Model (PyTorch)
def extract_s4d_eigenvalues():
    H = config["hidden_dim"]
    N = config["state_dim"]
    dt = config["timestep"]

    # Initialize S4D kernel
    s4d_kernel = S4DKernel(d_model=H, N=N)

    # Extract continuous-time eigenvalues
    A_real = -torch.exp(s4d_kernel.log_A_real)  # (H, N//2)
    A_imag = s4d_kernel.A_imag  # (H, N//2)

    # First H, N//2 values and their complex conjugates
    Lambda = A_real.detach().numpy() + 1j * A_imag.detach().numpy()
    Lambda = Lambda.reshape(-1)  # Flatten to 1D

    # Create conjugates
    Lambda_full = np.concatenate([Lambda, Lambda.conjugate()])

    # Discrete-time eigenvalues
    Lambda_disc = np.exp(Lambda_full * dt)

    return Lambda_disc


# 3. Initialize S5 Model (JAX/Equinox)
def extract_s5_eigenvalues():
    H = config["hidden_dim"]
    N = config["state_dim"]
    dt = config["timestep"]

    # Initialize S5 model
    model_key = random.PRNGKey(0)
    s5_model = S5(
        key=model_key,
        num_blocks=config["num_blocks"],
        N=config["input_dim"],
        ssm_size=N,
        ssm_blocks=1,
        H=H,
        output_dim=config["output_dim"],
    )

    # Extract eigenvalues
    Lambda_re = np.array(
        s5_model.__getstate__()["blocks"][0].__getstate__()["ssm"].Lambda_re
    )
    Lambda_im = np.array(
        s5_model.__getstate__()["blocks"][0].__getstate__()["ssm"].Lambda_im
    )

    # Continuous-time eigenvalues
    Lambda = Lambda_re + 1j * Lambda_im

    # Discrete-time eigenvalues
    Lambda_disc = np.exp(Lambda * dt)

    return Lambda_disc


# 4. Initialize LRU Model (JAX/Equinox)
def extract_lru_eigenvalues():
    H = config["hidden_dim"]
    N = config["state_dim"]
    dt = config["timestep"]

    # Initialize LRU model with explicit ranges for eigenvalues
    model_key = random.PRNGKey(0)
    lru_model = LRU(
        num_blocks=config["num_blocks"],
        data_dim=config["input_dim"],
        N=N,
        H=H,
        output_dim=config["output_dim"],
        classification=False,
        output_step=1,
        r_min=0.9,  # Minimum radius
        r_max=1.0,  # Maximum radius (stable if < 1)
        max_phase=6.38 / 10,  # Full circle in radians
        key=model_key,
    )

    # Extract continuous eigenvalues from first block
    block = lru_model.__getstate__()["blocks"][0].__getstate__()["lru"]
    nu_log = block.nu_log
    theta_log = block.theta_log

    # Compute Lambda (diagonal of state matrix in continuous time)
    Lambda = -np.exp(np.array(nu_log)) + 1j * np.exp(np.array(theta_log))

    # Create full spectrum with conjugates
    Lambda_full = np.concatenate([Lambda, np.conj(Lambda)])

    # Discrete-time eigenvalues
    Lambda_disc = np.exp(Lambda_full)

    return Lambda_disc


# 5. Initialize LinOSS IMEX Model (JAX/Equinox)
def extract_linoss_imex_eigenvalues():
    H = config["hidden_dim"]
    N = config["state_dim"]

    # Initialize LinOSS model with IMEX
    model_key = random.PRNGKey(0)
    linoss_model = LinOSS(
        num_blocks=config["num_blocks"],
        N=config["input_dim"],
        ssm_size=N,
        H=H,
        output_dim=config["output_dim"],
        classification=False,
        output_step=1,
        discretization="IMEX",
        key=model_key,
    )

    # Extract A diagonal from the first block
    block = linoss_model.__getstate__()["blocks"][0].__getstate__()["ssm"]
    A_diag = np.array(block.A_diag)

    # Apply activation as in forward pass
    A_diag = np.maximum(A_diag, 0)  # ReLU activation

    # Get step parameter and apply sigmoid as in forward pass
    steps = np.array(block.steps)
    steps = 1.0 / (1.0 + np.exp(-steps))  # sigmoid activation

    # For IMEX discretization, according to the equations in your prompt
    # We need the eigenvalues of the M matrix:
    # M_IMEX = [I      -Δt·A]
    #          [ΔtI  I-Δt²·A]

    # For a diagonal A, we can calculate the eigenvalues directly
    # The eigenvalues of 2x2 block matrix with diagonal A can be calculated
    # for each diagonal element of A independently

    # Calculate eigenvalues for each element in A_diag
    Lambda_disc = np.zeros(A_diag.shape[0], dtype=np.complex128)

    for i, (a, dt) in enumerate(zip(A_diag, steps)):
        # M = [[1, -dt*a], [dt, 1-dt²*a]]
        # Calculate eigenvalues of 2x2 matrix
        # Using the formula for eigenvalues of a 2x2 matrix
        trace = 2 - dt**2 * a
        det = 1 - dt**2 * a
        discriminant = trace**2 - 4 * det

        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            lambda1 = (trace + sqrt_disc) / 2
            lambda2 = (trace - sqrt_disc) / 2
            # Use the eigenvalue with larger magnitude
            Lambda_disc[i] = lambda1 if abs(lambda1) > abs(lambda2) else lambda2
        else:
            sqrt_disc = np.sqrt(-discriminant)
            real_part = trace / 2
            imag_part = sqrt_disc / 2
            Lambda_disc[i] = complex(real_part, imag_part)

    # For visualization purposes, create conjugate pairs
    Lambda_full = np.concatenate([Lambda_disc, np.conj(Lambda_disc)])

    return Lambda_full


# 6. Initialize LinOSS IM Model
def extract_linoss_im_eigenvalues():
    H = config["hidden_dim"]
    N = config["state_dim"]

    # Initialize LinOSS model with IM
    model_key = random.PRNGKey(1)  # Different seed from IMEX
    linoss_model = LinOSS(
        num_blocks=config["num_blocks"],
        N=config["input_dim"],
        ssm_size=N,
        H=H,
        output_dim=config["output_dim"],
        classification=False,
        output_step=1,
        discretization="IM",  # Use IM discretization
        key=model_key,
    )

    # Extract A diagonal from the first block
    block = linoss_model.__getstate__()["blocks"][0].__getstate__()["ssm"]
    A_diag = np.array(block.A_diag)

    # Apply activation as in forward pass
    A_diag = np.maximum(A_diag, 0)  # ReLU activation

    # Get step parameter and apply sigmoid
    steps = np.array(block.steps)
    steps = 1.0 / (1.0 + np.exp(-steps))  # sigmoid activation

    # For IM discretization, according to the equations in your prompt
    # We have these parameters:
    # schur_comp = 1. / (1. + step ** 2. * A_diag)
    # M_IM_11 = 1. - step ** 2. * A_diag * schur_comp
    # M_IM_12 = -1. * step * A_diag * schur_comp
    # M_IM_21 = step * schur_comp
    # M_IM_22 = schur_comp

    # Calculate eigenvalues for each element in A_diag
    Lambda_disc = np.zeros(A_diag.shape[0], dtype=np.complex128)

    for i, (a, dt) in enumerate(zip(A_diag, steps)):
        # Calculate components
        schur_comp = 1.0 / (1.0 + dt**2 * a)
        m11 = 1.0 - dt**2 * a * schur_comp
        m12 = -1.0 * dt * a * schur_comp
        m21 = dt * schur_comp
        m22 = schur_comp

        # Calculate eigenvalues of the 2x2 matrix
        trace = m11 + m22
        det = m11 * m22 - m12 * m21
        discriminant = trace**2 - 4 * det

        if discriminant >= 0:
            sqrt_disc = np.sqrt(discriminant)
            lambda1 = (trace + sqrt_disc) / 2
            lambda2 = (trace - sqrt_disc) / 2
            # Use the eigenvalue with larger magnitude
            Lambda_disc[i] = lambda1 if abs(lambda1) > abs(lambda2) else lambda2
        else:
            sqrt_disc = np.sqrt(-discriminant)
            real_part = trace / 2
            imag_part = sqrt_disc / 2
            Lambda_disc[i] = complex(real_part, imag_part)

    # For visualization purposes, create conjugate pairs
    Lambda_full = np.concatenate([Lambda_disc, np.conj(Lambda_disc)])

    return Lambda_full


# Generate and save plots for all models
def main():
    # Extract eigenvalues for all models
    eigenvalues = {}

    try:
        eigenvalues["S4"] = extract_s4_eigenvalues()
    except Exception as e:
        print(f"Error analyzing S4 model: {e}")

    try:
        eigenvalues["S4D"] = extract_s4d_eigenvalues()
    except Exception as e:
        print(f"Error analyzing S4D model: {e}")

    try:
        eigenvalues["S5"] = extract_s5_eigenvalues()
    except Exception as e:
        print(f"Error analyzing S5 model: {e}")

    try:
        eigenvalues["LRU"] = extract_lru_eigenvalues()
    except Exception as e:
        print(f"Error analyzing LRU model: {e}")

    try:
        eigenvalues["LinOSS-IMEX"] = extract_linoss_imex_eigenvalues()
    except Exception as e:
        print(f"Error analyzing LinOSS-IMEX model: {e}")

    try:
        eigenvalues["LinOSS-IM"] = extract_linoss_im_eigenvalues()
    except Exception as e:
        print(f"Error analyzing LinOSS-IM model: {e}")

    # Create combined visualization
    visualize_discrete_eigenvalues(
        eigenvalues,
        "eigenspectra_plots/ssm_discrete_eigenvalues.png",
        "Comparison of SSM Discrete Eigenspectra",
    )

    print("Eigenvalue plot generated successfully!")


if __name__ == "__main__":
    main()

# Foundational SSM imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D
import os

WRITE_FIG_DIR = "/cs/student/projects1/ml/2024/mlaimon/UCL-ML-Thesis/Writeup/figures"


def aggregate_bin_label_results(
    trial_info, target_vel, pred_vel, skip_timesteps=56, dt=0.005
):
    target_vel_sliced = target_vel[:, skip_timesteps:, :]
    pred_vel_sliced = pred_vel[:, skip_timesteps:, :]
    target_pos = np.cumsum(target_vel_sliced * dt, axis=1)
    pred_pos = np.cumsum(pred_vel_sliced * dt, axis=1)

    rows = []
    for split in ["train", "val"]:
        for angle_bin_label in trial_info.angle_bin_label.unique():
            filter_idx = trial_info[
                (trial_info["angle_bin_label"] == angle_bin_label)
                & (trial_info["split"] == split)
            ].index.values
            if len(filter_idx) == 0:
                continue
            r2 = r2_score(
                target_vel_sliced[filter_idx].reshape(-1, 2),
                pred_vel_sliced[filter_idx].reshape(-1, 2),
            )
            r2_varW = r2_score(
                target_vel_sliced[filter_idx].reshape(-1, 2),
                pred_vel_sliced[filter_idx].reshape(-1, 2),
                multioutput="variance_weighted",
            )
            r2_x = r2_score(
                np.atleast_2d(target_vel_sliced[filter_idx].reshape(-1, 2))[:, 0],
                np.atleast_2d(pred_vel_sliced[filter_idx].reshape(-1, 2))[:, 0],
            )
            r2_y = r2_score(
                np.atleast_2d(target_vel_sliced[filter_idx].reshape(-1, 2))[:, 1],
                np.atleast_2d(pred_vel_sliced[filter_idx].reshape(-1, 2))[:, 1],
            )

            rows.append(
                {
                    "split": split,
                    "angle_bin_label": angle_bin_label,
                    "target_pos_mean": np.mean(target_pos[filter_idx], axis=0),
                    "target_pos_std": np.std(target_pos[filter_idx], axis=0),
                    "pred_pos_mean": np.mean(pred_pos[filter_idx], axis=0),
                    "pred_pos_std": np.std(pred_pos[filter_idx], axis=0),
                    "error": np.std(
                        target_pos[filter_idx] - pred_pos[filter_idx], axis=0
                    ),
                    "r2": r2,
                    "r2_varW": r2_varW,
                    "r2_x": r2_x,
                    "r2_y": r2_y,
                }
            )

    results_df = pd.DataFrame(rows).sort_values(by=["split", "angle_bin_label"])
    return results_df


def plot_pred_vs_targets_by_angle_bin(results_df, suptitle=None, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    splits = ["train", "val"]
    colors = plt.cm.hsv(np.linspace(0, 1, len(results_df.angle_bin_label.unique())))

    for i, split in enumerate(splits):
        ax = axs[i]
        split_df = results_df[results_df["split"] == split]
        for j, row in split_df.reset_index().iterrows():
            label = row["angle_bin_label"]
            target_mean = row["target_pos_mean"]
            pred_mean = row["pred_pos_mean"]
            r2_val = row["r2_varW"]
            error = row["error"]

            ax.plot(
                target_mean[:, 0],
                target_mean[:, 1],
                linestyle="--",
                color=colors[j],
                label=f"{label} (R²={r2_val:.2f})",
            )
            ax.plot(pred_mean[:, 0], pred_mean[:, 1], linestyle="-", color=colors[j])

            # ax.fill_betweenx(
            #     pred_mean[:, 1],
            #     pred_mean[:, 0] - error[:, 0],
            #     pred_mean[:, 0] + error[:, 0],
            #     color=colors[j], alpha=0.1
            # )

        ax.set_title(f"{(split if split == 'train' else 'test').capitalize()} Set")
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")

        # Custom legend for line styles
        handles, labels_ = ax.get_legend_handles_labels()
        custom_lines = [
            Line2D([0], [0], color="black", linestyle="--", label="Target (dotted)"),
            Line2D([0], [0], color="black", linestyle="-", label="Prediction (solid)"),
        ]
        ax.legend(handles=handles + custom_lines, fontsize=8, loc="upper left")
    plt.suptitle(
        suptitle if suptitle else "Predicted vs Target Positions by Angle Bin",
        fontsize=16,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig


def plot_best_worst_trials(
    pred_vel,
    target_vel,
    mask,
    trial_info,
    split_key: str = "val",
    N: int = 4,
    title_prefix: str = "RTT",
    save_prefix: str = "rtt",
    write_dir: str = WRITE_FIG_DIR,
    title_cols=None,
):
    """Plot best and worst N single-trial predictions vs targets for a dataset.

    Args:
        pred_vel: array (trials, timesteps, 2) of predictions
        target_vel: array (trials, timesteps, 2) of targets
        mask: array (trials, timesteps) boolean/int mask of valid timesteps
        trial_info: DataFrame containing a 'split' column
        split_key: which split to filter trials by (e.g., 'val' or 'test')
        N: number of best/worst trials to display
        title_prefix: string prefix for figure suptitle
        save_prefix: filename prefix for saving PDFs
        write_dir: output directory for saving figures
        title_cols: optional str or list[str] of trial_info columns to append to each subplot title
    """

    # ignore warmup timesteps
    mask = mask.copy()
    mask[:, :60] = 0

    # filter by split
    split_idx = trial_info["split"] == split_key
    pv = pred_vel[split_idx.values]
    tv = target_vel[split_idx.values]
    mk = mask[split_idx.values]
    filtered_info = trial_info.loc[split_idx].reset_index(drop=True)
    # normalize title_cols into a list
    if title_cols is None:
        title_cols_list = []
    elif isinstance(title_cols, str):
        title_cols_list = [title_cols]
    else:
        title_cols_list = list(title_cols)

    # compute trial-wise R2 with masking
    r2_scores = []
    for i in range(pv.shape[0]):
        targ_m = np.where(mk[i, :, None], tv[i], 0)
        pred_m = np.where(mk[i, :, None], pv[i], 0)
        r2 = r2_score(targ_m, pred_m, multioutput="variance_weighted")
        r2_scores.append((i, r2))
    # fast lookup
    r2_lookup = {i: r for i, r in r2_scores}

    # sort and select
    r2_sorted = sorted(r2_scores, key=lambda x: x[1])
    worst = [idx for idx, _ in r2_sorted[:N]]
    best = [idx for idx, _ in reversed(r2_sorted[-N:])]

    # shared legend entries
    custom_lines = [
        Line2D([0], [0], color="blue", lw=2),
        Line2D([0], [0], color="green", lw=2, linestyle="dashed"),
    ]

    def _trial_bounds(mk_row, warmup=60, max_len=None):
        v = np.asarray(mk_row, dtype=bool).copy()
        v[:warmup] = False
        if v.any():
            idx = np.flatnonzero(v)
            t0, t1 = int(idx[0]), int(idx[-1] + 1)
            return t0, t1
        # fallback to a fixed window after warmup (match previous 150-step window if possible)
        t0 = warmup
        max_len = int(max_len if max_len is not None else len(mk_row))
        t1 = min(warmup + 150, max_len)
        return t0, t1

    # best
    fig, axs = plt.subplots(2, N, figsize=(2.5 * N, 4), sharex=True)
    for i, tr in enumerate(best):
        t0, t1 = _trial_bounds(mk[tr], warmup=60, max_len=pv.shape[1])
        time_s = np.arange(t0, t1) * 0.005  # s
        axs[0, i].plot(time_s, pv[tr, t0:t1, 0], label="pred", color="blue")
        axs[0, i].plot(
            time_s, tv[tr, t0:t1, 0], label="target", linestyle="dashed", color="green"
        )
        # build optional title suffix from trial_info
        suffix_parts = []
        for col in title_cols_list:
            if col in filtered_info.columns:
                val = filtered_info.iloc[tr][col]
                suffix_parts.append(f"{col.title()}={val}")
        suffix = ("\n" + " | ".join(suffix_parts)) if suffix_parts else ""
        axs[0, i].set_title(
            f"#{i + 1}: {r'$R^2$'} = {r2_lookup[tr]:.3f}{suffix}", fontsize=10
        )
        axs[1, i].plot(time_s, pv[tr, t0:t1, 1], label="pred", color="blue")
        axs[1, i].plot(
            time_s, tv[tr, t0:t1, 1], label="target", linestyle="dashed", color="green"
        )
    for ax in axs[1, :]:
        ax.set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("X Velocity (au)")
    axs[1, 0].set_ylabel("Y Velocity (au)")
    fig.legend(
        custom_lines,
        ["Prediction", "Target"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )
    fig.suptitle(
        f"{title_prefix} Best Single Trial Predictions vs Targets", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(
        os.path.join(
            write_dir, f"{save_prefix}_single_trial_pred_vs_target_best_{N}.pdf"
        )
    )

    # worst
    fig2, axs2 = plt.subplots(2, N, figsize=(2.5 * N, 4), sharex=True)
    for i, tr in enumerate(worst):
        t0, t1 = _trial_bounds(mk[tr], warmup=60, max_len=pv.shape[1])
        time_s = np.arange(t0, t1) * 0.005  # s
        axs2[0, i].plot(time_s, pv[tr, t0:t1, 0], label="pred", color="blue")
        axs2[0, i].plot(
            time_s, tv[tr, t0:t1, 0], label="target", linestyle="dashed", color="green"
        )
        # build optional title suffix from trial_info
        suffix_parts = []
        for col in title_cols_list:
            if col in filtered_info.columns:
                val = filtered_info.iloc[tr][col]
                suffix_parts.append(f"{col.title()}={val}")
        suffix = ("\n" + " | ".join(suffix_parts)) if suffix_parts else ""
        axs2[0, i].set_title(
            f"#{i + 1}: {r'$R^2$'} = {r2_lookup[tr]:.3f}{suffix}", fontsize=10
        )
        axs2[1, i].plot(time_s, pv[tr, t0:t1, 1], label="pred", color="blue")
        axs2[1, i].plot(
            time_s, tv[tr, t0:t1, 1], label="target", linestyle="dashed", color="green"
        )
    for ax in axs2[1, :]:
        ax.set_xlabel("Time (s)")
    axs2[0, 0].set_ylabel("X Velocity (au)")
    axs2[1, 0].set_ylabel("Y Velocity (au)")
    fig2.legend(
        custom_lines,
        ["Prediction", "Target"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )
    fig2.suptitle(
        f"{title_prefix} Worst Single Trial Predictions vs Targets", fontsize=14
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(
        os.path.join(
            write_dir, f"{save_prefix}_single_trial_pred_vs_target_worst_{N}.pdf"
        )
    )

    plt.show()

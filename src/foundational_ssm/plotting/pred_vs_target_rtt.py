
# Foundational SSM imports
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D


def aggregate_bin_label_results(trial_info, target_vel, pred_vel, skip_timesteps=56, dt=0.005):

    target_vel_sliced = target_vel[:, skip_timesteps:, :]
    pred_vel_sliced = pred_vel[:, skip_timesteps:, :]
    target_pos = np.cumsum(target_vel_sliced * dt, axis=1)
    pred_pos = np.cumsum(pred_vel_sliced * dt, axis=1)

    rows = []
    for split in ['train', 'val']:
        for angle_bin_label in trial_info.angle_bin_label.unique():
            filter_idx = trial_info[(trial_info['angle_bin_label'] == angle_bin_label) & (trial_info["split"] == split)].index.values
            if len(filter_idx) == 0:
                continue
            r2 = r2_score(target_vel_sliced[filter_idx].reshape(-1, 2), pred_vel_sliced[filter_idx].reshape(-1, 2))
            r2_varW = r2_score(target_vel_sliced[filter_idx].reshape(-1, 2), pred_vel_sliced[filter_idx].reshape(-1, 2), multioutput='variance_weighted')
            r2_x = r2_score(np.atleast_2d(target_vel_sliced[filter_idx].reshape(-1, 2))[:, 0], np.atleast_2d(pred_vel_sliced[filter_idx].reshape(-1, 2))[:, 0])
            r2_y = r2_score(np.atleast_2d(target_vel_sliced[filter_idx].reshape(-1, 2))[:, 1], np.atleast_2d(pred_vel_sliced[filter_idx].reshape(-1, 2))[:, 1])

            rows.append({
                'split': split,
                'angle_bin_label': angle_bin_label,
                'target_pos_mean': np.mean(target_pos[filter_idx], axis=0),
                'target_pos_std': np.std(target_pos[filter_idx], axis=0),
                'pred_pos_mean': np.mean(pred_pos[filter_idx], axis=0),
                'pred_pos_std': np.std(pred_pos[filter_idx], axis=0),
                'error': np.std(target_pos[filter_idx] - pred_pos[filter_idx], axis=0),
                'r2': r2,
                'r2_varW': r2_varW,
                'r2_x': r2_x,
                'r2_y': r2_y
            })

    results_df = pd.DataFrame(rows).sort_values(by=['split', 'angle_bin_label'])
    return results_df

def plot_pred_vs_targets_by_angle_bin(results_df, suptitle=None, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    splits = ['train', 'val']
    colors = plt.cm.hsv(np.linspace(0, 1, len(results_df.angle_bin_label.unique())))

    for i, split in enumerate(splits):
        ax = axs[i]
        split_df = results_df[results_df['split'] == split]
        for j, row in split_df.reset_index().iterrows():
            label = row['angle_bin_label']
            target_mean = row['target_pos_mean']
            pred_mean = row['pred_pos_mean']
            r2_val = row['r2']
            error = row['error']
            
            ax.plot(target_mean[:, 0], target_mean[:, 1], linestyle='--', color=colors[j], label=f'{label} (R²={r2_val:.2f})')
            ax.plot(pred_mean[:, 0], pred_mean[:, 1], linestyle='-', color=colors[j])

            ax.fill_betweenx(
                pred_mean[:, 1],
                pred_mean[:, 0] - error[:, 0],
                pred_mean[:, 0] + error[:, 0],
                color=colors[j], alpha=0.1
            )

        ax.set_title(f"{split.capitalize()} Set")
        ax.set_xlabel("Position X")
        ax.set_ylabel("Position Y")

        # Custom legend for line styles
        handles, labels_ = ax.get_legend_handles_labels()
        custom_lines = [
            Line2D([0], [0], color='black', linestyle='--', label='Target (dotted)'),
            Line2D([0], [0], color='black', linestyle='-', label='Prediction (solid)')
        ]
        ax.legend(handles=handles + custom_lines, fontsize=8, loc='best')
    plt.suptitle(suptitle if suptitle else "Predicted vs Target Positions by Angle Bin", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    return fig
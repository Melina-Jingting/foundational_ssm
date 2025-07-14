import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from foundational_ssm.data_utils.loaders import get_brainset_train_val_loaders

config_path = "configs/pretrain.yaml"
cfg = OmegaConf.load(config_path)

train_dataset, train_loader, val_dataset, val_loader = get_brainset_train_val_loaders(
    train_config=cfg.train_dataset,
    val_config=cfg.val_dataset,
    **cfg.dataloader,
)

# Get sampling intervals for train and val
train_intervals = train_dataset.get_sampling_intervals()
val_intervals = val_dataset.get_sampling_intervals()

# Plot for the first 5 sessions
sessions = list(train_intervals.keys())[:5]
fig, axes = plt.subplots(len(sessions), 1, figsize=(12, 2 * len(sessions)), sharex=True)

if len(sessions) == 1:
    axes = [axes]

for idx, session in enumerate(sessions):
    ax = axes[idx]
    # Plot train intervals (blue)
    for interval in train_intervals[session]:
        start = float(interval.start)
        end = float(interval.end)
        ax.plot([start, end], [1, 1], color='blue', linewidth=8, solid_capstyle='butt', label='train' if idx == 0 else "")
    # Plot val intervals (green)
    for interval in val_intervals.get(session, []):
        start = float(interval.start)
        end = float(interval.end)
        ax.plot([start, end], [1.2, 1.2], color='green', linewidth=8, solid_capstyle='butt', label='val' if idx == 0 else "")
    ax.set_title(f"Session: {session}")
    ax.set_yticks([])
    ax.set_ylabel("intervals")
    ax.legend(loc='upper right')
    ax.grid(True, axis='x')

plt.xlabel("Time")
plt.tight_layout()
plt.show()


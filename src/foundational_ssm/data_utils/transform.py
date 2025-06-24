
from typing import Dict, Any

import numpy as np
import torch

from foundational_ssm.constants.dataset_info import GROUP_DIMS, parse_session_id
from foundational_ssm.data_utils.spikes import bin_spikes, smooth_spikes



def _ensure_dim(arr: np.ndarray, target_dim: int, *, axis: int = 1) -> np.ndarray:
    """Crop or zero-pad *arr* along *axis* to match *target_dim*.

    This is a thin wrapper around :pymod:`numpy` slicing and :func:`numpy.pad` that
    avoids several conditional blocks in the main routine.
    """
    current_dim = arr.shape[axis]
    if current_dim == target_dim:
        return arr  # nothing to do
    if current_dim > target_dim:
        # Crop
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(None, target_dim)
        return arr[tuple(slicer)]
    # Pad (current_dim < target_dim)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_dim - current_dim)
    return np.pad(arr, pad_width, mode="constant")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def transform_neural_behavior_sample(
    data: Any,
    *,
    sampling_rate: int = 100,
    num_timesteps: int = 100,
    kern_sd_ms: int = 40,
    bin_width: int = 5,
) -> Dict[str, torch.Tensor | str]:
    """Convert a *temporaldata* sample to a dictionary of Torch tensors.

    The function takes care of binning & smoothing spikes, cropping/padding neural
    and behavioural features to a globally consistent dimensionality that depends
    on the *(dataset, subject, task)* triple.

    Parameters
    ----------
    data: temporaldata.Data
        Sample returned by **torch-brain**/**temporaldata**.
    sampling_rate: int, default=100
        Target sampling rate *Hz* used for binning.
    num_timesteps: int, default=100
        Length of the temporal window after binning.
    kern_sd_ms: int, default=40
        Standard deviation of the Gaussian kernel (in ms) for smoothing spikes.
    bin_width: int, default=5
        Width of the time bins (in ms) used by :func:`smooth_spikes`.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with keys ``neural_input``, ``behavior_input``, ``session_id``
        and ``subject_id``.
    """

    # ------------------------------------------------------------------
    # 1. Bin + smooth spikes
    # ------------------------------------------------------------------
    unit_ids = data.units.id
    binned_spikes = bin_spikes(
        spikes=data.spikes,
        num_units=len(unit_ids),
        bin_size=1 / sampling_rate,
        num_bins=num_timesteps,
    )  # shape: (units, timesteps)

    # Transpose so that time is first → (timesteps, units)
    binned_spikes = binned_spikes.T

    smoothed_spikes = smooth_spikes(
        binned_spikes,
        kern_sd_ms=kern_sd_ms,
        bin_width=bin_width,
    )

    # ------------------------------------------------------------------
    # 2. Prepare behaviour signal (cursor velocity)
    # ------------------------------------------------------------------
    behavior_input = data.cursor.vel  # np.ndarray, (timesteps?, features)

    # Crop/pad time dimension *first* so that we can treat both signals equally
    if behavior_input.shape[0] > num_timesteps:
        behavior_input = behavior_input[:num_timesteps]
    elif behavior_input.shape[0] < num_timesteps:
        behavior_input = np.pad(
            behavior_input,
            ((0, num_timesteps - behavior_input.shape[0]), (0, 0)),
            mode="constant",
        )

    # ------------------------------------------------------------------
    # 3. Align channel dimensions based on (dataset, subject, task)
    # ------------------------------------------------------------------
    dataset, subject, task = parse_session_id(data.session.id)

    try:
        neural_dim, output_dim = GROUP_DIMS[(dataset, subject, task)]
    except KeyError as exc:
        raise ValueError(
            f"Group {(dataset, subject, task)} not found in predefined GROUP_DIMS"
        ) from exc

    # Crop / pad along *unit* axis
    smoothed_spikes = _ensure_dim(smoothed_spikes, neural_dim, axis=1)
    # Behavioural features (axis=1 again because time already first)
    behavior_input = _ensure_dim(behavior_input, output_dim, axis=1)

    # ------------------------------------------------------------------
    # 4. Pack into torch tensors
    # ------------------------------------------------------------------
    return {
        "neural_input": torch.as_tensor(smoothed_spikes, dtype=torch.float32),
        "behavior_input": torch.as_tensor(behavior_input, dtype=torch.float32),
        "dataset_group_key": f"{dataset}-{subject}-{task}"
    } 
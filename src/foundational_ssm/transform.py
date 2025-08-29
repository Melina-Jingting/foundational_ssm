from typing import Dict, Any, Tuple

import numpy as np
import torch
import re

from foundational_ssm.constants import (
    DATASET_GROUP_TO_IDX,
    MAX_NEURAL_UNITS,
    MAX_BEHAVIOR_DIM,
    DATASET_IDX_TO_STD
)
from foundational_ssm.spikes import bin_spikes, smooth_spikes


def parse_session_id(session_id: str) -> Tuple[str, str, str]:
    patterns = {
        "churchland_shenoy_neural_2012": re.compile(r"([^/]+)/([^_]+)_[0-9]+_(.+)"),
        "flint_slutzky_accurate_2012": re.compile(r"([^/]+)/monkey_([^_]+)_e1_(.+)"),
        "odoherty_sabes_nonhuman_2017": re.compile(r"([^/]+)/([^_]+)_[0-9]{8}_[0-9]+"),
        "pei_pandarinath_nlb_2021": re.compile(r"([^/]+)/([^_]+)_(.+)"),
        "perich_miller_population_2018": re.compile(r"([^/]+)/([^_]+)_[0-9]+_(.+)"),
    }

    dataset = session_id.split('/')[0]
    if dataset not in patterns:
        raise ValueError(f"Unknown dataset: {dataset}")

    match = patterns[dataset].match(session_id)
    if not match:
        raise ValueError(f"Could not parse session_id: {session_id!r}")

    if dataset == "odoherty_sabes_nonhuman_2017":
        # Always assign task as 'random_target_reaching'
        _, subject = match.groups()
        return dataset, subject, "random_target_reaching"
    elif dataset == "flint_slutzky_accurate_2012":
        # task is always 'center_out_reaching'
        _, subject, _ = match.groups()
        return dataset, subject, "center_out_reaching"
    else:
        return match.groups()

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



def transform_brainsets_regular_time_series_smoothed(
    data: Any,
    *,
    max_neural_units: int = MAX_NEURAL_UNITS,
    sampling_rate: int = 200,
    kern_sd_ms: int = 20,
) -> Dict[str, torch.Tensor | str]:
    """
    Convert a temporaldata sample to a dictionary of Torch tensors with standardized dimensions.

    This function bins and smooths spike data, prepares and cleans behavioral signals,
    removes time steps with invalid (NaN/inf) behavioral values, and ensures that both
    neural and behavioral features are padded or cropped to globally consistent dimensions
    based on the (dataset, subject, task) triple. The output is a dictionary of torch tensors
    ready for model input, including group index information.

    Parameters
    ----------
    data: temporaldata.Data
        Sample returned by **torch-brain**/**temporaldata**.
    sampling_rate: int, default=100
        Target sampling rate *Hz* used for binning.
    sampling_window_ms: int, default=1000   
        Length of the temporal window after binning.
    kern_sd_ms: int, default=20
        Standard deviation of the Gaussian kernel (in ms) for smoothing spikes.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with keys ``neural_input``, ``behavior_input``, ``session_id``
        and ``subject_id``.
    """
    try:
        bin_size_ms = int(1000 / sampling_rate)
        
        # ------------------------------------------------------------------
        # 1. Get smoothed spikes
        # ------------------------------------------------------------------
        if hasattr(data, "smoothed_binned_spikes"):
            smoothed_spikes = data.smoothed_binned_spikes.data
        else:
            binned_spikes, _ = data.spikes.get_regular_time_series_array(
                sampling_rate=sampling_rate,
                raw_array_name="unit_index",
                is_index=True,
            )  # shape: (timesteps, units)

            smoothed_spikes = smooth_spikes(
                binned_spikes,
                kern_sd_ms=kern_sd_ms,
                bin_size_ms=bin_size_ms,
                time_axis=0,
            )

        # ------------------------------------------------------------------
        # 2. Prepare behaviour signal (cursor velocity)
        # ------------------------------------------------------------------
        if hasattr(data, "vel_regular"):
            behavior_input = data.vel_regular.data
        else:
            if data.session.id.startswith("pei_pandarinath_nlb_2021"):
                behavior_input, _ = data.hand.get_regular_time_series_array(
                    sampling_rate=sampling_rate,
                    raw_array_name="vel",
                )  # np.ndarray, (timesteps, features)
            else:
                behavior_input, _ = data.cursor.get_regular_time_series_array(
                    sampling_rate=sampling_rate,
                    raw_array_name="vel",
                )  # np.ndarray, (timesteps, features)

        # ------------------------------------------------------------------
        # 3. Drop timesteps where behavior has inf/NaN values
        # ------------------------------------------------------------------
        # Check for inf/NaN in behavior_input
        valid_mask = ~(np.isinf(behavior_input).any(axis=1) | np.isnan(behavior_input).any(axis=1))
        if not valid_mask.all():
            behavior_input = behavior_input[valid_mask]
            smoothed_spikes = smoothed_spikes[valid_mask]
        
        # ------------------------------------------------------------------
        # 4. Align channel dimensions based on (dataset, subject, task) and normalize
        # ------------------------------------------------------------------
        dataset, subject, task = parse_session_id(data.session.id)
        group_tuple = (dataset, subject, task)
        try:
            group_idx = DATASET_GROUP_TO_IDX[group_tuple]
        except:
            group_idx = 9

        match = re.findall(r'\d+', data.session.id.split('/')[1])
        session_date = int(''.join(match)) if len(match) > 0 else 0

        smoothed_spikes = _ensure_dim(smoothed_spikes, max_neural_units, axis=1)
        
        behavior_input = behavior_input / DATASET_IDX_TO_STD[group_idx] 
        behavior_input = _ensure_dim(behavior_input, MAX_BEHAVIOR_DIM, axis=1)
        
        # ------------------------------------------------------------------
        # 5. Pack into torch tensors
        # ------------------------------------------------------------------
        return {
            "neural_input": torch.as_tensor(smoothed_spikes, dtype=torch.float32),
            "behavior_input": torch.as_tensor(behavior_input, dtype=torch.float32),
            "dataset_group_idx": torch.as_tensor(group_idx, dtype=torch.int32),
            "session_date": torch.as_tensor(session_date, dtype=torch.int32)
        }
    except:
        raise
    
    
    
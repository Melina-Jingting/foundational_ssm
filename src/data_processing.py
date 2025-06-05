import numpy as np
import re

def bin_spikes(spikes, num_units, bin_size, right=True, num_bins=None):
    """
    Bins spike timestamps into a 2D array: [num_units x num_bins].
    """
    rate = 1 / bin_size  # avoid precision issues
    binned_spikes = np.zeros((num_units, num_bins))
    bin_index = np.floor((spikes.timestamps) * rate).astype(int)
    np.add.at(binned_spikes, (spikes.unit_index, bin_index), 1)
    return binned_spikes

def map_binned_features_to_global(
    session_binned_features,
    session_unit_id_strings,
    max_global_units=192
):
    """
    Map session-specific binned neural features to a global, padded array.

    Args:
        session_binned_features (np.ndarray): (num_bins, num_session_units)
        session_unit_id_strings (list/np.ndarray): Unit ID strings for the session (len = num_session_units)
        max_global_units (int): Output array's second dimension size

    Returns:
        np.ndarray: (num_bins, max_global_units)
    """
    if not (isinstance(session_binned_features, np.ndarray) and session_binned_features.ndim == 2):
        raise ValueError("session_binned_features must be 2D np.ndarray")
    if len(session_unit_id_strings) != session_binned_features.shape[1]:
        raise ValueError("session_unit_id_strings length must match num_session_units")

    global_binned = np.zeros((session_binned_features.shape[0], max_global_units), dtype=session_binned_features.dtype)
    for i, unit_str in enumerate(session_unit_id_strings):
        m = re.search(r'elec(\d+)', unit_str)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < max_global_units:
                global_binned[:, idx] = session_binned_features[:, i]
    return global_binned
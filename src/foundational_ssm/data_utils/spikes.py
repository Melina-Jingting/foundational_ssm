import numpy as np
import re


def bin_spikes(spikes, num_units, sampling_rate, right=True, num_bins=None):
    """
    Bins spike timestamps into a 2D array: [num_units x num_bins].
    """
    binned_spikes = np.zeros((num_units, num_bins))
    bin_index = np.floor((spikes.timestamps) * sampling_rate).astype(int)
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

def gaussian_window(M, std, sym=True):
    """
    Create a Gaussian window without using scipy.signal.
    
    Parameters:
    -----------
    M : int
        Number of points in the window
    std : float
        Standard deviation of the Gaussian
    sym : bool, optional
        Whether the window is symmetric (default=True)
        
    Returns:
    --------
    numpy.ndarray
        Gaussian window
    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1)
    
    if sym:
        n = np.arange(0, M) - (M - 1) / 2
    else:
        n = np.arange(0, M)
    
    sig2 = 2 * std * std
    w = np.exp(-n * n / sig2)
    
    return w

def smooth_spikes(spike_data, kern_sd_ms=40, bin_size_ms=5):
    """
    Apply Gaussian smoothing to spike data.
    
    Parameters:
    -----------
    spike_data : numpy.ndarray
        Spike data array to smooth, with time along axis 1
    kern_sd_ms : float, optional
        Standard deviation of Gaussian kernel in milliseconds, default 40ms
    bin_size_ms : float, optional
        Width of time bins in milliseconds, default 5ms
        
    Returns:
    --------
    numpy.ndarray
        Smoothed spike data with same shape as input
    """
    # Compute kernel standard deviation in bins
    kern_sd = int(round(kern_sd_ms / bin_size_ms))
    
    # Create Gaussian window
    window = gaussian_window(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    
    # Define convolution function
    filt = lambda x: np.convolve(x, window, 'same')
    
    # Apply along time axis
    smoothed_data = np.apply_along_axis(filt, 1, spike_data)
    
    return smoothed_data

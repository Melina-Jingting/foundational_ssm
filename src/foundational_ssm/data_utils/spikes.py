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

def smooth_spikes(spike_data, kern_sd_ms=40, bin_size_ms=5, time_axis=0):
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
    smoothed_data = np.apply_along_axis(filt, time_axis, spike_data)
    
    return smoothed_data

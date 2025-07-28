import h5py

def h5_to_dict(h5obj):
    """Recursive function that reads HDF5 file to dict

    Parameters
    ----------
    h5obj : h5py.File or h5py.Group
        File or group object to load into a dict
    
    Returns
    -------
    dict of np.array
        Dict mapping h5obj keys to arrays
        or other dicts
    """
    data_dict = {}
    for key in h5obj.keys():
        if isinstance(h5obj[key], h5py.Group):
            data_dict[key] = h5_to_dict(h5obj[key])
        else:
            data_dict[key] = h5obj[key][()]
    return data_dict
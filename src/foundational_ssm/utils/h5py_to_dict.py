import h5py


def h5_to_dict(h5_path_or_obj):
    """Recursive function that reads HDF5 file or group into a dict.

    Parameters
    ----------
    h5_path_or_obj : str or h5py.File or h5py.Group
        Path to the HDF5 file, or an h5py.File object, or an h5py.Group object
        to load into a dict.

    Returns
    -------
    dict of np.array
        Dict mapping h5obj keys to NumPy arrays
        or other dicts for nested groups.
    """

    if isinstance(h5_path_or_obj, str):
        with h5py.File(h5_path_or_obj, "r") as h5obj:
            return _h5_to_dict_recursive(h5obj)
    elif isinstance(h5_path_or_obj, (h5py.File, h5py.Group)):
        return _h5_to_dict_recursive(h5_path_or_obj)
    else:
        raise TypeError(
            "Input must be a string (file path), h5py.File, or h5py.Group object."
        )


def _h5_to_dict_recursive(h5obj):
    """
    Helper recursive function to read HDF5 group/file contents into a dictionary.
    This function is called by h5_to_dict after handling the initial input type.
    """
    data_dict = {}
    for key in h5obj.keys():
        item = h5obj[key]
        if isinstance(item, h5py.Group):
            data_dict[key] = _h5_to_dict_recursive(item)
        else:
            data_dict[key] = item[()]
    return data_dict

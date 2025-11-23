class NLBDictDataset(torch.utils.data.Dataset):
    def __init__(self, spikes, behavior, held_out_flags):
        self.spikes = spikes
        self.behavior = behavior
        self.held_out_flags = held_out_flags

    def __len__(self):
        return len(self.spikes)

    def __getitem__(self, i):
        return {
            "neural_input": self.spikes[i],
            "behavior_input": self.behavior[i],
            "held_out": self.held_out_flags[i],
        }


# Unified function to get held_out flags for each trial (0 for held-in, 1 for held-out)
def get_held_out_flags(trial_info, dataset, task=None):
    if dataset == "mc_maze" and task == "center_out_reaching":
        heldin_types = set(MC_MAZE_CONFIG.CENTER_OUT_HELD_IN_TRIAL_TYPES)
        return [0 if t in heldin_types else 1 for t in trial_info["trial_type"]]
    else:
        return [0] * len(trial_info)


def get_nlb_train_val_loaders(
    dataset="mc_rtt",
    task=None,
    holdout_angles=False,
    batch_size=256,
    data_root=NLB_DATA_ROOT,
    num_workers=8,
):
    """
    Loads NLB-processed data and returns train/val datasets and DataLoaders with batches matching get_train_val_loaders.

    Args:
        dataset (str): Dataset name.
        task (str): Task name. Some datasets have multiple tasks, e.g. mc_maze
        processed_data_path (str): Path to the .h5 file with NLB data.
        trial_info_path (str): Path to the .csv file with trial split info.
        batch_size (int): Batch size for DataLoaders.
        group_key (tuple): (dataset, subject, task) tuple for DATASET_GROUP_TO_IDX.
        collate_fn (callable): Collate function for DataLoader.
        device (str or torch.device, optional): If set, moves tensors to device.

    Returns:
        train_dataset, train_loader, val_dataset, val_loader
    """

    def jax_collate_fn(batch):
        """
        Collate function that converts all torch.Tensors in a batch (dict or list of dicts)
        to numpy arrays, recursively.
        """
        collated = default_collate(batch)
        return tree_map(
            lambda x: x.numpy() if isinstance(x, torch.Tensor) else x, collated
        )

    collate_fn = jax_collate_fn
    task_config = NLB_CONFIGS[dataset]

    data_path = os.path.join(data_root, task_config.H5_FILE_NAME)
    trial_info_path = os.path.join(data_root, task_config.TRIAL_INFO_FILE_NAME)
    with h5py.File(data_path, "r") as h5file:
        dataset_dict = h5_to_dict(h5file)
    trial_info = pd.read_csv(trial_info_path)
    trial_info = trial_info[trial_info["split"].isin(["train", "val"])]

    if task == "center_out_reaching":
        trial_info = trial_info[
            trial_info["trial_version"] == MC_MAZE_CONFIG.TASK_TO_TRIAL_VERSION[task]
        ]

    if holdout_angles:
        # Use held_out flags to filter trial_info for training
        held_out_flags = get_held_out_flags(trial_info, dataset, task)
        # Only keep held-in trials for training (held_out == 0)
        train_mask = (trial_info["split"] != "train") | (np.array(held_out_flags) == 0)
        trial_info = trial_info[train_mask]

    min_idx = trial_info["trial_id"].min()
    trial_info["trial_id"] = trial_info["trial_id"] - min_idx

    # Concatenate heldin and heldout spikes
    spikes = np.concatenate(
        [dataset_dict["train_spikes_heldin"], dataset_dict["train_spikes_heldout"]],
        axis=2,
    )

    # Use bin_size_ms=5 to match NLB binning
    smoothed_spikes = smooth_spikes(spikes, kern_sd_ms=20, bin_size_ms=5)
    behavior = dataset_dict["train_behavior"]
    # smoothed_spikes = _ensure_dim(smoothed_spikes, MAX_NEURAL_INPUT_DIM, axis=2)
    behavior = _ensure_dim(behavior, MAX_BEHAVIOR_DIM, axis=2)

    trial_info = trial_info.sort_values("trial_id").reset_index(drop=True)
    train_mask = trial_info[trial_info["split"] == "train"].index
    val_mask = trial_info[trial_info["split"] == "val"].index

    train_spikes = smoothed_spikes[train_mask]
    train_behavior = behavior[train_mask]
    val_spikes = smoothed_spikes[val_mask]
    val_behavior = behavior[val_mask]

    train_held_out = get_held_out_flags(
        trial_info[trial_info["split"] == "train"], dataset, task
    )
    val_held_out = get_held_out_flags(
        trial_info[trial_info["split"] == "val"], dataset, task
    )

    train_dataset = NLBDictDataset(train_spikes, train_behavior, train_held_out)
    val_dataset = NLBDictDataset(val_spikes, val_behavior, val_held_out)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(pad_collate, fixed_seq_len=600),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(pad_collate, fixed_seq_len=1000),
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_dataset, train_loader, val_dataset, val_loader

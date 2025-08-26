from functools import partial
from torch.utils.data import DataLoader

from foundational_ssm.constants import DATA_ROOT, MAX_NEURAL_UNITS, DATASET_GROUP_INFO
from foundational_ssm.dataset import TorchBrainDataset
from foundational_ssm.transform import transform_brainsets_regular_time_series_smoothed, parse_session_id
from foundational_ssm.collate import pad_collate
import foundational_ssm.samplers as samplers
import numpy as np

def get_brainset_data_loader(
    dataset_args,
    dataloader_args,
    sampler,
    split = None,
    sampler_args = {},
    sampling_rate = 200,
    prepend_history = 0,
    data_root = DATA_ROOT,
):
    dataset = TorchBrainDataset(
        root=data_root,                # root directory where .h5 files are found
        **dataset_args,
        split=split
    )

    sampling_intervals = dataset.get_sampling_intervals()
    sampler_cls = getattr(samplers, sampler)
    sampler = sampler_cls(
        sampling_intervals=sampling_intervals,
        **(sampler_args or {}),
        prepend_history=prepend_history
    )
    max_neural_units = int(np.max( [DATASET_GROUP_INFO[parse_session_id(k)]["max_num_units"] for k in sampling_intervals.keys()]))
    dataset.transform = partial(transform_brainsets_regular_time_series_smoothed, sampling_rate=sampling_rate, max_neural_units=max_neural_units)
    total_window_length = sampler_args.get('window_length', sampler_args.get('max_window_length', 1)) + prepend_history  # Default to 1 if not provided
    loader = DataLoader(
        dataset=dataset,      # dataset
        sampler=sampler,      # sampler
        collate_fn=partial(pad_collate, fixed_seq_len=int(total_window_length*sampling_rate)),         # the collator
        pin_memory=True,
        **dataloader_args
    )
    return dataset, loader, max_neural_units

def get_brainset_train_val_loaders(
    dataset_args,
    train_loader_cfg,
    val_loader_cfg,
    prepend_history=0,
    data_root=DATA_ROOT,
):
    train_dataset, train_loader, train_max_neural_units = get_brainset_data_loader( **train_loader_cfg, dataset_args=dataset_args, split='train', prepend_history=prepend_history, data_root=data_root)
    val_dataset, val_loader, val_max_neural_units = get_brainset_data_loader( **val_loader_cfg, dataset_args=dataset_args, split='val_trial', prepend_history=prepend_history, data_root=data_root)
    max_neural_units = max(train_max_neural_units, val_max_neural_units)
    return train_dataset, train_loader, val_dataset, val_loader, max_neural_units
    



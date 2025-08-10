from functools import partial
from torch.utils.data import DataLoader

from foundational_ssm.constants import DATA_ROOT
from foundational_ssm.dataset import TorchBrainDataset
from foundational_ssm.transform import transform_brainsets_regular_time_series_smoothed
from foundational_ssm.collate import pad_collate
import foundational_ssm.samplers as samplers


def get_brainset_data_loader(
    dataset_args,
    sampler,
    sampler_args,
    dataloader_args,
    # window_length,
    sampling_rate,
    dataset_cfg,
    data_root=DATA_ROOT
):
    window_length = sampler_args.get('window_length', 1)  # Default to 1 if not provided
    dataset = TorchBrainDataset(
        root=data_root,                # root directory where .h5 files are found
        transform=partial(transform_brainsets_regular_time_series_smoothed, sampling_rate=sampling_rate),
        **dataset_args,
        config=dataset_cfg,  # configuration for the dataset
    )

    sampling_intervals = dataset.get_sampling_intervals()
    sampler_cls = getattr(samplers, sampler)
    sampler = sampler_cls(
        sampling_intervals=sampling_intervals,
        **(sampler_args or {})
    )
    
    loader = DataLoader(
        dataset=dataset,      # dataset
        sampler=sampler,      # sampler
        collate_fn=partial(pad_collate, fixed_seq_len=int(window_length*sampling_rate)),         # the collator
        pin_memory=True,
        **dataloader_args
    )
    return dataset, loader

def get_brainset_train_val_loaders(
    train_loader_cfg,
    val_loader_cfg,
    dataset_cfg,
    data_root=DATA_ROOT,
):
    train_dataset, train_loader = get_brainset_data_loader( **train_loader_cfg, dataset_cfg=dataset_cfg, data_root=data_root)
    val_dataset, val_loader = get_brainset_data_loader( **val_loader_cfg, dataset_cfg=dataset_cfg, data_root=data_root)
    return train_dataset, train_loader, val_dataset, val_loader
    



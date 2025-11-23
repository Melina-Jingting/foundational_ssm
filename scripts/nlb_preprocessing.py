import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from nlb_tools.nwb_interface import NWBDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NLB datasets into padded train/val tensors."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to a YAML/JSON configuration describing the datasets to preprocess.",
    )
    parser.add_argument(
        "--prepend-duration",
        type=int,
        default=299,
        help="Milliseconds of history to prepend to each trial before padding (default: 299).",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def dict_to_h5(tensor_dict: Dict[str, Any], output_h5_file: str) -> None:
    import h5py  # Imported lazily to avoid mandatory dependency for non-preprocessing workflows

    os.makedirs(os.path.dirname(output_h5_file), exist_ok=True)
    with h5py.File(output_h5_file, "w") as f:
        for key, value in tensor_dict.items():
            if torch.is_tensor(value):
                tensor = value.detach().cpu()
                if tensor.dtype == torch.bool:
                    data_to_save = tensor.to(torch.uint8).numpy()
                else:
                    data_to_save = tensor.numpy()
            else:
                data_to_save = value

            f.create_dataset(key, data=data_to_save)


def build_split_batch(
    nwb_dataset: NWBDataset,
    splits: Sequence[str] | str,
    *,
    prepend_duration: int,
    dataset_group_idx: int,
    bin_width: Optional[int],
    dtype_neural: np.dtype,
    dtype_behavior: np.dtype,
    train_std: Optional[np.ndarray],
    behavior_attribute: str,
    start_field: str,
    end_field: str,
) -> Dict[str, Any]:
    from typing import cast

    if bin_width is not None:
        nwb_dataset.resample(bin_width)

    trials = nwb_dataset.trial_info
    split_list = [splits] if isinstance(splits, str) else list(splits)
    mask_split = trials["split"].isin(split_list)
    trials_sel = trials[mask_split].reset_index(drop=True)
    if trials_sel.empty:
        return {
            "neural_input": torch.empty(0),
            "behavior_input": torch.empty(0),
            "mask": torch.empty(0),
            "dataset_group_idx": torch.empty(0),
        }

    behavior_arr = cast(
        pd.DataFrame, getattr(nwb_dataset.data, behavior_attribute)
    ).to_numpy(dtype=dtype_behavior)
    neural_arr = nwb_dataset.data.spikes.to_numpy(dtype=dtype_neural)
    time_index = nwb_dataset.data.index.values

    starts = trials_sel[start_field] - pd.Timedelta(prepend_duration, "ms")
    ends = trials_sel[end_field]

    start_idx = np.searchsorted(time_index, starts.values)
    end_idx = np.searchsorted(time_index, ends.values, side="right")

    T = behavior_arr.shape[0]
    start_idx = np.clip(start_idx, 0, T - 1)
    end_idx = np.clip(end_idx, 1, T)

    lengths = (end_idx - start_idx).astype(int)
    max_len = lengths.max()

    beh_dim = behavior_arr.shape[1] if behavior_arr.ndim > 1 else 1
    neu_dim = neural_arr.shape[1] if neural_arr.ndim > 1 else 1
    n_trials = len(trials_sel)

    padded_behavior = np.zeros((n_trials, max_len, beh_dim), dtype=behavior_arr.dtype)
    padded_neural = np.zeros((n_trials, max_len, neu_dim), dtype=neural_arr.dtype)
    mask = np.zeros((n_trials, max_len), dtype=bool)

    for i, (s_idx, e_idx) in enumerate(zip(start_idx, end_idx)):
        length = int(e_idx - s_idx)
        if length <= 0:
            continue
        padded_behavior[i, :length, :] = behavior_arr[s_idx:e_idx]
        padded_neural[i, :length, :] = neural_arr[s_idx:e_idx]
        mask[i, :length] = True

    behavior_std = None
    if bin_width is None or bin_width <= 0:
        offset = 0
    else:
        offset = prepend_duration // bin_width + 1

    if offset < mask.shape[1]:
        valid_mask = mask[:, offset:].reshape(-1)
        if valid_mask.any():
            flat_beh = padded_behavior[:, offset:, :].reshape(-1, beh_dim)
            valid_beh = flat_beh[valid_mask]
            if isinstance(splits, str) and splits == "train" and train_std is None:
                behavior_std = np.nanstd(valid_beh, axis=0, ddof=0)
                behavior_std[behavior_std == 0] = 1.0
                padded_behavior = padded_behavior / behavior_std[None, None, :]
            elif train_std is not None:
                ts = np.asarray(train_std)
                if ts.size != beh_dim:
                    raise ValueError(
                        f"train_std length {ts.size} does not match behavior dim {beh_dim}"
                    )
                padded_behavior = padded_behavior / ts[None, None, :]
            elif train_std is None:
                behavior_std = np.std(valid_beh, axis=0, ddof=0)
                behavior_std[behavior_std == 0] = 1.0
                padded_behavior = padded_behavior / behavior_std[None, None, :]

    out: Dict[str, Any] = {
        "neural_input": torch.from_numpy(padded_neural),
        "behavior_input": torch.from_numpy(padded_behavior),
        "mask": torch.from_numpy(mask),
        "dataset_group_idx": torch.full(
            (n_trials,), dataset_group_idx, dtype=torch.int8
        ),
        "trial_idx": torch.from_numpy(trials_sel["trial_id"].values.astype(np.int32)),
    }
    if behavior_std is not None:
        out["behavior_std"] = behavior_std
    return out


def target_change_trialize(
    nwb_dataset: NWBDataset,
    *,
    target_attribute: str,
    split_fraction: float,
) -> pd.DataFrame:
    target_df = getattr(nwb_dataset.data, target_attribute)
    if target_df is None:
        raise ValueError(f"Dataset is missing target attribute '{target_attribute}'.")

    filled = target_df.fillna(-1000)
    has_change = filled.diff(axis=0).any(axis=1)
    change_nan = nwb_dataset.data.loc[has_change].isna().any(axis=1)
    drop_trial = (
        change_nan
        | change_nan.shift(1, fill_value=True)
        | change_nan.shift(-1, fill_value=True)
    )[:-1]

    change_times = nwb_dataset.data.index[has_change]
    if len(change_times) < 2:
        raise ValueError("Insufficient target changes to define trials.")

    start_times = change_times[:-1][~drop_trial]
    end_times = change_times[1:][~drop_trial]

    prior_times = start_times - pd.Timedelta(1, "ms")
    start_pos = target_df.loc[prior_times].to_numpy().tolist()
    target_pos = target_df.loc[start_times].to_numpy().tolist()
    reach_dist = (
        target_df.loc[end_times - pd.Timedelta(1, "ms")].to_numpy()
        - target_df.loc[prior_times].to_numpy()
    )
    reach_angle = np.degrees(np.arctan2(reach_dist[:, 1], reach_dist[:, 0]))

    trial_info = pd.DataFrame(
        {
            "trial_id": np.arange(len(start_times)),
            "start_time": start_times,
            "end_time": end_times,
            "duration": (end_times - start_times).total_seconds(),
            "start_pos": start_pos,
            "target_pos": target_pos,
            "reach_dist_x": reach_dist[:, 0],
            "reach_dist_y": reach_dist[:, 1],
            "reach_angle": reach_angle,
        }
    )

    trial_info["split"] = "train"
    train_count = int(len(trial_info) * split_fraction)
    train_count = np.clip(train_count, 1, len(trial_info) - 1)
    trial_info.loc[trial_info.index[train_count:], "split"] = "val"

    return trial_info


def ensure_split_column(
    trial_info: pd.DataFrame,
    *,
    split_fraction: float,
) -> pd.DataFrame:
    if "split" in trial_info.columns:
        return trial_info

    trial_info = trial_info.copy()
    trial_info["split"] = "train"
    train_count = int(len(trial_info) * split_fraction)
    train_count = np.clip(train_count, 1, len(trial_info) - 1)
    trial_info.loc[trial_info.index[train_count:], "split"] = "val"
    return trial_info


def preprocess_dataset(
    dataset_cfg: Dict[str, Any],
    base_folder: Path,
    prepend_duration: int,
) -> Tuple[str, str]:
    name = dataset_cfg["name"]
    subpath = dataset_cfg["subpath"]
    dataset_group_idx = int(dataset_cfg.get("dataset_group_idx", 0))
    behavior_attribute = dataset_cfg.get("behavior_attribute", "cursor_vel")
    start_field = dataset_cfg.get("start_field", "start_time")
    end_field = dataset_cfg.get("end_field", "end_time")
    bin_width = dataset_cfg.get("bin_width", 5)
    dtype_neural = np.dtype(dataset_cfg.get("dtype_neural", np.int8))
    dtype_behavior = np.dtype(dataset_cfg.get("dtype_behavior", np.float32))
    split_fraction = float(dataset_cfg.get("split_fraction", 0.7))
    output_prefix = dataset_cfg.get("output_prefix", f"{name}")
    split_heldout = bool(dataset_cfg.get("split_heldout", False))
    save_trial_info = bool(dataset_cfg.get("save_trial_info", True))
    target_attr = dataset_cfg.get("target_attribute", "target_pos")

    raw_data_path = base_folder / "raw" / "dandi" / subpath
    raw_data_path = raw_data_path.resolve()
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")

    nwb_dataset = NWBDataset(str(raw_data_path), split_heldout=split_heldout)

    trialization = dataset_cfg.get("trialization")
    if trialization == "target_change":
        trial_info = target_change_trialize(
            nwb_dataset,
            target_attribute=target_attr,
            split_fraction=split_fraction,
        )
        nwb_dataset.trial_info = trial_info

    nwb_dataset.resample(bin_width)
    nwb_dataset.trial_info = ensure_split_column(
        nwb_dataset.trial_info,
        split_fraction=split_fraction,
    )

    train_out = build_split_batch(
        nwb_dataset,
        "train",
        prepend_duration=prepend_duration,
        dataset_group_idx=dataset_group_idx,
        bin_width=bin_width,
        dtype_neural=dtype_neural,
        dtype_behavior=dtype_behavior,
        train_std=None,
        behavior_attribute=behavior_attribute,
        start_field=start_field,
        end_field=end_field,
    )

    train_std = train_out.get("behavior_std")
    train_batch = {k: v for k, v in train_out.items() if k != "behavior_std"}

    val_out = build_split_batch(
        nwb_dataset,
        "val",
        prepend_duration=prepend_duration,
        dataset_group_idx=dataset_group_idx,
        bin_width=bin_width,
        dtype_neural=dtype_neural,
        dtype_behavior=dtype_behavior,
        train_std=train_std,
        behavior_attribute=behavior_attribute,
        start_field=start_field,
        end_field=end_field,
    )
    val_batch = {k: v for k, v in val_out.items() if k != "behavior_std"}

    processed_root = dataset_cfg.get("output_dir")
    if processed_root is None:
        processed_root = base_folder / "processed" / "nlb"
    else:
        processed_root = Path(processed_root)

    os.makedirs(processed_root, exist_ok=True)

    train_path = processed_root / f"{output_prefix}_prepend_train.h5"
    val_path = processed_root / f"{output_prefix}_prepend_val.h5"

    dict_to_h5(train_batch, str(train_path))
    dict_to_h5(val_batch, str(val_path))

    if save_trial_info:
        trial_info_path = processed_root / f"{output_prefix}_trial_info.csv"
        nwb_dataset.trial_info.to_csv(trial_info_path)

    return str(train_path), str(val_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.input_file)

    dataset_root = cfg.get(
        "dataset_folder",
        "/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/",
    )
    base_folder = Path(dataset_root)
    datasets_cfg = cfg.get("datasets")
    if not isinstance(datasets_cfg, Iterable):
        raise ValueError("Configuration must provide a 'datasets' list.")

    for dataset_cfg in datasets_cfg:
        if not isinstance(dataset_cfg, dict):
            raise ValueError("Each dataset configuration must be a mapping.")
        name = dataset_cfg.get("name", "<unknown>")
        print(f"Processing dataset '{name}'...")
        train_path, val_path = preprocess_dataset(
            dataset_cfg,
            base_folder,
            args.prepend_duration,
        )
        print(f"  Train tensors saved to: {train_path}")
        print(f"  Val tensors saved to:   {val_path}")


if __name__ == "__main__":
    main()

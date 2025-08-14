
from foundational_ssm.constants import DATA_ROOT
from omegaconf import OmegaConf
from foundational_ssm.dataset import TorchBrainDataset
from foundational_ssm.loaders import transform_brainsets_regular_time_series_smoothed
import numpy as np
from tqdm import tqdm
import h5py
from temporaldata import Data
from foundational_ssm.transform import smooth_spikes
from temporaldata import RegularTimeSeries, Interval
import os

def main():
    config_path = "/nfs/ghome/live/mlaimon/foundational_ssm/configs/dataset/cs.yaml"
    # cfg = OmegaConf.load(config_path)

    all_dataset = TorchBrainDataset(
        root=DATA_ROOT,                # root directory where .h5 files are found
        config=config_path,                 # or a config for multi-session training / more complex configs
        keep_files_open=False,
        lazy=True,
    )

    sampling_rate = 200
    bin_size_ms = 1000/sampling_rate

    recording_ids = all_dataset.get_session_ids()
    all_dataset._close_open_files()
    for recording_id in tqdm(recording_ids):
        input_path = f"{DATA_ROOT}/{recording_id}.h5"
        output_path = f"{DATA_ROOT}/v2/{recording_id}.h5"

        with h5py.File(input_path) as f:
            recording_data = Data.from_hdf5(f, lazy=True)
            if "smoothed_binned_spikes" in recording_data.keys():
                print("smoothed_binned_spikes in recording_data.keys()")

        with h5py.File(input_path) as f:
            recording_data = Data.from_hdf5(f, lazy=False)
            regular_hand, regular_hand_times = recording_data.cursor.get_regular_time_series_array(sampling_rate, "vel")
            vel_regular = RegularTimeSeries(
                data=regular_hand,
                sampling_rate=sampling_rate,
                domain=Interval(start=min(regular_hand_times), end=max(regular_hand_times)),
            )

            binned_spikes, binned_spike_times = recording_data.spikes.get_regular_time_series_array(sampling_rate, "unit_index", is_index=True)
            smoothed_spikes = smooth_spikes(binned_spikes, kern_sd_ms=20, bin_size_ms=5, time_axis=0)

            smoothed_binned_spikes = RegularTimeSeries(
                data=smoothed_spikes,
                sampling_rate=sampling_rate,
                domain=Interval(start=min(binned_spike_times), end=max(binned_spike_times)),
            )
            recording_data.vel_regular = vel_regular
            recording_data.smoothed_binned_spikes = smoothed_binned_spikes

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as f:
            recording_data.to_hdf5(f)

if __name__ == "__main__":
    main()
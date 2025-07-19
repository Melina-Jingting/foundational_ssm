import os
from tqdm import tqdm
import h5py
from foundational_ssm.data_utils.dataset import TorchBrainDataset
from foundational_ssm.constants import DATA_ROOT
from foundational_ssm.data_utils.spikes import smooth_spikes
from temporaldata import Data, RegularTimeSeries, Interval

config_path = "/cs/student/projects1/ml/2024/mlaimon/foundational_ssm/configs/dataset/all_brainsets.yaml"


def process_recording(recording_id):
    sampling_rate = 200
    bin_size_ms = 1000 / sampling_rate
    h5_path = os.path.join(DATA_ROOT, f"{recording_id}.h5")
    with h5py.File(h5_path, "r+") as f:
        recording_data = Data.from_hdf5(f, lazy=False)
        regular_hand, regular_hand_times = recording_data.cursor.get_regular_time_series_array(sampling_rate, "vel")
        vel_regular = RegularTimeSeries(
            data=regular_hand,
            sampling_rate=sampling_rate,
            domain=Interval(start=min(regular_hand_times), end=max(regular_hand_times)),
        )
        binned_spikes, binned_spike_times = recording_data.spikes.get_regular_time_series_array(sampling_rate, "unit_index", is_index=True)
        smoothed_spikes = smooth_spikes(binned_spikes, kern_sd_ms=20, bin_size_ms=int(bin_size_ms), time_axis=0)
        smoothed_binned_spikes = RegularTimeSeries(
            data=smoothed_spikes,
            sampling_rate=sampling_rate,
            domain=Interval(start=min(binned_spike_times), end=max(binned_spike_times)),
        )
        recording_data.vel_regular = vel_regular
        recording_data.smoothed_binned_spikes = smoothed_binned_spikes
        recording_data.to_hdf5(f)
    return recording_id


def main():
    all_dataset = TorchBrainDataset(
        root=os.path.join(DATA_ROOT),
        config=config_path,
        keep_files_open=False,
        lazy=True,
    )
    session_ids = all_dataset.get_session_ids()
    for rid in tqdm(session_ids, desc="Processing sessions"):
        try:
            process_recording(rid)
        except Exception as e:
            print(f"Error processing {rid}: {e}")


if __name__ == "__main__":
    main()

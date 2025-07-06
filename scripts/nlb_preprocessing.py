from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
from nlb_tools.evaluation import evaluate
import os
import pandas as pd
import numpy as np

dataset_folder = '/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/' 

datasets = [
    # {'name':'mc_maze', 'subpath':'./000128/sub-Jenkins/'},
    {'name':'mc_rtt', 'subpath':'./000129/sub-Indy/'},
    # {'name':'area2_bump', 'subpath':'./000127/sub-Han/'},
    # {'name':'dmfc_rsg', 'subpath':'./000130/sub-Haydn/'},
    # {'name':'mc_maze_large', 'subpath':'./000138/sub-Jenkins/'},
]

for d in datasets: 
    dataset_name = d['name']
    dataset_subpath = d['subpath']

    raw_data_path = os.path.join(dataset_folder, 'raw', 'dandi', dataset_subpath) 
    processed_data_folder = os.path.join(dataset_folder, 'processed', 'nlb')
    processed_data_path = os.path.join(processed_data_folder, dataset_name + '.h5')
    trial_info_path = os.path.join(processed_data_folder, dataset_name + '.csv')

    if not os.path.exists(processed_data_folder):
        print(f"Creating directory: {processed_data_folder}")
        os.makedirs(processed_data_folder, exist_ok=True)

    nwb_dataset = NWBDataset(raw_data_path)
    bin_width = 5
    nwb_dataset.resample(bin_width)
    suffix = '' if (bin_width == 5) else f'_{int(round(bin_width))}'

    train_trial_split = ['train', 'val', 'test']
    if dataset_name == 'mc_rtt':
        has_change = nwb_dataset.data.target_pos.fillna(-1000).diff(axis=0).any(axis=1) # filling NaNs with arbitrary scalar to treat as one block
        change_nan = nwb_dataset.data[has_change].isna().any(axis=1)
        drop_trial = (change_nan | change_nan.shift(1, fill_value=True) | change_nan.shift(-1, fill_value=True))[:-1]
        change_times = nwb_dataset.data.index[has_change]
        start_times = change_times[:-1][~drop_trial]
        end_times = change_times[1:][~drop_trial]
        target_pos = nwb_dataset.data.target_pos.loc[start_times].to_numpy().tolist()
        reach_dist = nwb_dataset.data.target_pos.loc[end_times - pd.Timedelta(1, 'ms')].to_numpy() - nwb_dataset.data.target_pos.loc[start_times - pd.Timedelta(1, 'ms')].to_numpy()
        reach_angle = np.arctan2(reach_dist[:, 1], reach_dist[:, 0]) / np.pi * 180
        nwb_dataset.trial_info = pd.DataFrame({
            'trial_id': np.arange(len(start_times)),
            'start_time': start_times,
            'end_time': end_times,
            'target_pos': target_pos,
            'reach_dist_x': reach_dist[:, 0],
            'reach_dist_y': reach_dist[:, 1],
            'reach_angle': reach_angle,
        })

    # Based on original NLB splits
    train_dict = make_train_input_tensors(nwb_dataset, dataset_name=dataset_name, trial_split=train_trial_split, save_file=True, save_path=processed_data_path, include_forward_pred=True, include_behavior=True)
    trial_info = nwb_dataset.trial_info
    trial_info.to_csv(trial_info_path)

from typing import Dict, Tuple, Any
import re
import numpy as np
import jax.numpy as jnp


DATA_ROOT = "../data/foundational_ssm/processed"
# dataset, subject, task -> info
DATASET_GROUP_INFO: Dict[Tuple[str, str, str], Dict[str, Any]] = {
    
    # Perich Miller Population 2018
    ("perich_miller_population_2018", "c", "center_out_reaching"): {
        "max_num_units": 352,
        "behavior_dim": 2,
        "train_duration": 44338.492,
        "min_behavior_sampling_rate": 0.0,
        "model_encoder_index": 0,
        "short_name": "pm_c_co",
        "variance": 35.11016
    },
    ("perich_miller_population_2018", "c", "random_target_reaching"): {
        "max_num_units": 87,
        "behavior_dim": 2,
        "train_duration": 18720.241,
        "min_behavior_sampling_rate": 0.01,
        "model_encoder_index": 1,
        "short_name": "pm_c_rt",
        "variance": 64.80464
    },
    ("perich_miller_population_2018", "m", "random_target_reaching"): {
        "max_num_units": 164,
        "behavior_dim": 2,
        "train_duration": 5094.218,
        "min_behavior_sampling_rate": 0.01,
        "model_encoder_index": 2,
        "short_name": "pm_m_rt",
        "variance": 36.15431
    },
    ("perich_miller_population_2018", "m", "center_out_reaching"): {
        "max_num_units": 158,
        "behavior_dim": 2,
        "train_duration": 32402.495,
        "min_behavior_sampling_rate": 0.01,
        "model_encoder_index": 3,
        "short_name": "pm_m_co",
        "variance": 27.485073
    },

    # Pei Pandarinath NLB 2021
    ("pei_pandarinath_nlb_2021", "jenkins", "maze"): {
        "max_num_units": 141,
        "behavior_dim": 2,
        "train_duration": 205.57,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 4,
        "short_name": "pp_j_mz",
        "variance": 28500.521
    },

    # O'Doherty Sabes Nonhuman 2017
    ("odoherty_sabes_nonhuman_2017", "indy", "random_target_reaching"): {
        "max_num_units": 464,
        "behavior_dim": 2,
        "train_duration": 20432.648,
        "min_behavior_sampling_rate": 0.004,
        "model_encoder_index": 5,
        "short_name": "os_i_rt",
        "variance": 5406.2188
    },
    ("odoherty_sabes_nonhuman_2017", "loco", "random_target_reaching"): {
        "max_num_units": 625,
        "behavior_dim": 2,
        "train_duration": 14189.179,
        "min_behavior_sampling_rate": 0.004,
        "model_encoder_index": 6,
        "short_name": "os_l_rt",
        "variance": 1178.2418
    },

    # Churchland Shenoy Neural 2012
    ("churchland_shenoy_neural_2012", "jenkins", "center_out_reaching"): {
        "max_num_units": 190,
        "behavior_dim": 2,
        "train_duration": 34656.524,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 7,
        "short_name": "cs_j_co",
        "variance": 17223.988
    },
    ("churchland_shenoy_neural_2012", "nitschke", "center_out_reaching"): {
        "max_num_units": 191,
        "behavior_dim": 2,
        "train_duration": 85741.222,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 8,
        "short_name": "cs_n_co",
        "variance": 16115.329
    },
    ("error_record", "error_record", "error_record"): {
        "max_num_units": 625,
        "behavior_dim": 2,
        "train_duration": 0,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 9,
        "short_name": "error",
        "variance": 9999999
    },
}



DATASET_GROUPS = list(DATASET_GROUP_INFO.keys())
DATASET_GROUP_TO_IDX = {group: idx for idx, group in enumerate(DATASET_GROUPS)}
DATASET_IDX_TO_GROUP = {idx: group for group, idx in DATASET_GROUP_TO_IDX.items()}
DATASET_IDX_TO_GROUP_SHORT = {idx: DATASET_GROUP_INFO[group]["short_name"] for group, idx in DATASET_GROUP_TO_IDX.items()}
DATASET_IDX_TO_STD = {idx: DATASET_GROUP_INFO[group]["variance"] ** 0.5 for group, idx in DATASET_GROUP_TO_IDX.items()}

def get_dataset_group_std():
    """Get the dataset group standard deviation array, creating it lazily to avoid JAX initialization at import time"""
    return jnp.array([
        DATASET_GROUP_INFO[group].get("variance", 1.0) ** 0.5
        for group in DATASET_GROUPS
    ])

MAX_NEURAL_UNITS = 625 
MAX_BEHAVIOR_DIM = 2

FINETUNING_DATASET_GROUPS = {
    # FINETUNING DATASETS
    ("perich_miller_population_2018", "t", "random_target_reaching"): {
        "max_num_units": 58,
        "behavior_dim": 2,
        "train_duration": 0,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 10,  #Placeholder to follow pretraining format
        "short_name": "pm_t_rt",
    },
    ("perich_miller_population_2018", "t", "center_out_reaching"): {
        "max_num_units": 58,
        "behavior_dim": 2,
        "train_duration": 0,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 11,  #Placeholder to follow pretraining format
        "short_name": "pm_t_co",
    },
}



# Pre-compiled regex to extract (dataset, subject, task) from a session id.
# Example id: "perich_miller_population_2018/c_20131003_center_out_reaching"



# DATASET_GROUP_DIMS: Dict[Tuple[str, str, str], Tuple[int, int]] = {
#     # Perich Miller 2018: Called reaching but the monkey is actually controlling a cursor with joystick. 
#     # Velocity and position are of the CURSOR's. 
#     ("perich_miller_population_2018", "c", "center_out_reaching"): (353, 2),    #0
#     ("perich_miller_population_2018", "c", "random_target_reaching"): (88, 2),  #1
#     ("perich_miller_population_2018", "m", "center_out_reaching"): (159, 2),    #2
#     ("perich_miller_population_2018", "m", "random_target_reaching"): (165, 2), #3
#     ("perich_miller_population_2018", "t", "center_out_reaching"): (65, 2),     #4
#     ("perich_miller_population_2018", "t", "random_target_reaching"): (73, 2),  #5
#     ("perich_miller_population_2018", "j", "center_out_reaching"): (38, 2),     #6
#     ('pei_pandarinath_nlb_2021', 'jenkins', 'maze'): (352, 2),
#     ('odoherty_sabes_nonhuman_2017', 'indy', 'random_target_reaching'): (464, 2),
#     ('odoherty_sabes_nonhuman_2017', 'loco', 'random_target_reaching'): (625, 2),
#     ('churchland_shenoy_neural_2012', 'jenkins', 'center_out_reaching'): (625, 2),
#     ('churchland_shenoy_neural_2012', 'nitschke', 'center_out_reaching'): (625, 2)
# }
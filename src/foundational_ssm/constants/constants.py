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
        "output_variance": [33.483936, 36.412506]
    },
    ("perich_miller_population_2018", "c", "random_target_reaching"): {
        "max_num_units": 87,
        "behavior_dim": 2,
        "train_duration": 18720.241,
        "min_behavior_sampling_rate": 0.01,
        "model_encoder_index": 1,
        "short_name": "pm_c_rt",
        "output_variance": [62.93111, 65.5517]
    },
    ("perich_miller_population_2018", "m", "random_target_reaching"): {
        "max_num_units": 164,
        "behavior_dim": 2,
        "train_duration": 5094.218,
        "min_behavior_sampling_rate": 0.01,
        "model_encoder_index": 2,
        "short_name": "pm_m_rt",
        "output_variance": [36.383457, 36.838387]
    },
    ("perich_miller_population_2018", "m", "center_out_reaching"): {
        "max_num_units": 158,
        "behavior_dim": 2,
        "train_duration": 32402.495,
        "min_behavior_sampling_rate": 0.01,
        "model_encoder_index": 3,
        "short_name": "pm_m_co",
        "output_variance": [24.888403, 29.958048]
    },

    # Pei Pandarinath NLB 2021
    ("pei_pandarinath_nlb_2021", "jenkins", "maze"): {
        "max_num_units": 141,
        "behavior_dim": 2,
        "train_duration": 205.57,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 4,
        "short_name": "pp_j_mz",
        "output_variance": [20047.146, 16873.195]
    },

    # O'Doherty Sabes Nonhuman 2017
    ("odoherty_sabes_nonhuman_2017", "indy", "random_target_reaching"): {
        "max_num_units": 464,
        "behavior_dim": 2,
        "train_duration": 20432.648,
        "min_behavior_sampling_rate": 0.004,
        "model_encoder_index": 5,
        "short_name": "os_i_rt",
        "output_variance": [6801.448, 3987.769]
    },
    ("odoherty_sabes_nonhuman_2017", "loco", "random_target_reaching"): {
        "max_num_units": 625,
        "behavior_dim": 2,
        "train_duration": 14189.179,
        "min_behavior_sampling_rate": 0.004,
        "model_encoder_index": 6,
        "short_name": "os_l_rt",
        "output_variance": [1375.8164, 1052.7667]
    },

    # Churchland Shenoy Neural 2012
    ("churchland_shenoy_neural_2012", "jenkins", "center_out_reaching"): {
        "max_num_units": 190,
        "behavior_dim": 2,
        "train_duration": 34656.524,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 7,
        "short_name": "cs_j_co",
        "output_variance": [19890.82 , 14408.727]
    },
    ("churchland_shenoy_neural_2012", "nitschke", "center_out_reaching"): {
        "max_num_units": 191,
        "behavior_dim": 2,
        "train_duration": 85741.222,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 8,
        "short_name": "cs_n_co",
        "output_variance": [17583.1  , 14569.994]
    },
    ("error_record", "error_record", "error_record"): {
        "max_num_units": 625,
        "behavior_dim": 2,
        "train_duration": 0,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 9,
        "short_name": "error",
        "output_variance": [1e10, 1e10]
    },
    ("perich_miller_population_2018", "t", "random_target_reaching"): {
        "max_num_units": 72,
        "behavior_dim": 2,
        "train_duration": 0,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 10,  #Placeholder to follow pretraining format
        "short_name": "pm_t_rt",
        "output_variance": [40.407703, 45.406548],
    },
    ("perich_miller_population_2018", "t", "center_out_reaching"): {
        "max_num_units": 72,
        "behavior_dim": 2,
        "train_duration": 0,
        "min_behavior_sampling_rate": 0.001,
        "model_encoder_index": 11,  #Placeholder to follow pretraining format
        "short_name": "pm_t_co",
        "output_variance": [21.67696 , 25.117935],
    }
}

DATASET_GROUPS = list(DATASET_GROUP_INFO.keys())
DATASET_GROUP_TO_IDX = {group: idx for idx, group in enumerate(DATASET_GROUPS)}
DATASET_IDX_TO_GROUP = {idx: group for group, idx in DATASET_GROUP_TO_IDX.items()}
DATASET_IDX_TO_GROUP_SHORT = {idx: DATASET_GROUP_INFO[group]["short_name"] for group, idx in DATASET_GROUP_TO_IDX.items()}
DATASET_IDX_TO_STD = {idx: np.array(DATASET_GROUP_INFO[group]["output_variance"]) ** 0.5 for group, idx in DATASET_GROUP_TO_IDX.items()}

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
    }
}

MC_RTT_VARIANCE = 1569.56
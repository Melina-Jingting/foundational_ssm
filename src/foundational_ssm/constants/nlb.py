

NLB_DATA_ROOT = "/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/processed/nlb"


class MC_MAZE_CONFIG:
    TASK_TO_TRIAL_VERSION = {
        'center_out_reaching': 0,
        'maze': 1,
        'maze_active_target': 2,
    }
    SUBJECT_NAME = 'j'
    H5_FILE_NAME = 'mc_maze.h5'
    TRIAL_INFO_FILE_NAME = 'mc_maze.csv'
    TASK_TO_DATASET_GROUP_KEY = {
        'center_out_reaching': ('nlb', 'j', 'center_out_reaching'),
        'maze': ('nlb', 'j', 'maze'),
        'maze_active_target': ('nlb', 'j', 'maze_active_target'),
    }
    TASK_TO_DATASET_GROUP_IDX = {
        'center_out_reaching': 7,
        'maze_active_target': 8,
    }
    CENTER_OUT_HELD_IN_TRIAL_TYPES = [6, 16, 22, 24, 36, # Bottom left
                           2, 12, 15, 18, 23, 28, 33, 34, # Bottom right 
                           7, 21, 25, 29 ]# Top

class MC_RTT_CONFIG:
    SUBJECT_NAME = 'Indy'
    H5_FILE_NAME = 'mc_rtt.h5'
    TRIAL_INFO_FILE_NAME = 'mc_rtt.csv'
    TASK_TO_DATASET_GROUP_KEY = {
        'random_target_reaching': ('nlb', 'Indy', 'random_target_reaching')
    }
    TASK_TO_DATASET_GROUP_IDX = {
        'random_target_reaching': 9
    }
    HELD_IN_REACH_ANGLE_RANGES = [[90+22.5, 90-22.5], [-45+22.5, -45-22.5], [-135+22.5, -135-22.5]]

NLB_CONFIGS = {
    'mc_maze': MC_MAZE_CONFIG,
    'mc_rtt': MC_RTT_CONFIG,
}






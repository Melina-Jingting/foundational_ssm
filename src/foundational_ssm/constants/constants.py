from typing import Dict, Tuple
import re

DATA_ROOT = "/nfs/ghome/live/mlaimon/data/foundational_ssm/processed"
DATASET_GROUP_DIMS: Dict[Tuple[str, str, str], Tuple[int, int]] = {
    # Perich Miller 2018: Called reaching but the monkey is actually controlling a cursor with joystick. 
    # Velocity and position are of the CURSOR's. 
    ("perich_miller_population_2018", "c", "center_out_reaching"): (353, 2),    #0
    ("perich_miller_population_2018", "c", "random_target_reaching"): (88, 2),  #1
    ("perich_miller_population_2018", "m", "center_out_reaching"): (159, 2),    #2
    ("perich_miller_population_2018", "m", "random_target_reaching"): (165, 2), #3
    ("perich_miller_population_2018", "t", "center_out_reaching"): (65, 2),     #4
    ("perich_miller_population_2018", "t", "random_target_reaching"): (73, 2),  #5
    ("perich_miller_population_2018", "j", "center_out_reaching"): (38, 2),     #6
    
    # NLB MC Maze: Monkey reaching out to targets in a circle or in a maze. 
    # Velocity and position are of the HAND's.
    ("nlb", "j", "center_out_reaching"): (182, 2),                              #7
    ("nlb", "j", "maze_active_target"): (182, 2),                                             #8
    ("nlb", "Indy", "random_target_reaching"): (182, 2),                               #9
}
MAX_NEURAL_INPUT_DIM = 353 
MAX_BEHAVIOR_INPUT_DIM = 2

def shorten_group_key(group_key: tuple) -> str:
    """
    Shorten a dataset group key tuple to a compact string.
    Example: ("perich_miller_population_2018", "c", "center_out_reaching") -> "pm_c_co"
    """
    dataset, subject, task = group_key
    # Shorten dataset
    if dataset == "perich_miller_population_2018":
        dataset_short = "pm"
    elif dataset == "nlb":
        dataset_short = "nlb"
    else:
        dataset_short = dataset[:2]
    # Shorten task
    if task == "center_out_reaching":
        task_short = "co"
    elif task == "random_target_reaching":
        task_short = "rt"
    elif task == "maze":
        task_short = "mz"
    elif task == "maze_active_target":
        task_short = "mat"
    else:
        task_short = task[:2]
    return f"{dataset_short}_{subject}_{task_short}"

DATASET_GROUPS = list(DATASET_GROUP_DIMS.keys())
DATASET_GROUP_TO_IDX = {group: idx for idx, group in enumerate(DATASET_GROUPS)}
DATASET_IDX_TO_GROUP = {idx: group for group, idx in DATASET_GROUP_TO_IDX.items()}
DATASET_IDX_TO_GROUP_SHORT = {idx: shorten_group_key(group) for group, idx in DATASET_GROUP_TO_IDX.items()}

# Pre-compiled regex to extract (dataset, subject, task) from a session id.
# Example id: "perich_miller_population_2018/c_20131003_center_out_reaching"
_SESSION_RE = re.compile(r"([^/]+)/([^_]+)_[^_]+_(.+)")

def parse_session_id(session_id: str) -> Tuple[str, str, str]:
    """Extract *(dataset, subject, task)* triple from the *session_id* string.

    Parameters
    ----------
    session_id: str
        Session identifier following the pattern
        ``{dataset}/{subject}_{date}_{task}``.

    Returns
    -------
    Tuple[str, str, str]
        Parsed *(dataset, subject, task)*. Raises *ValueError* if parsing fails.
    """
    match = _SESSION_RE.match(session_id)
    if match is None:
        raise ValueError(f"Could not parse session_id: {session_id!r}")
    return match.groups()  # type: ignore[return-value]

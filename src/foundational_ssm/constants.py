from typing import Dict, Tuple
import re

DATA_ROOT = "/cs/student/projects1/ml/2024/mlaimon/data/foundational_ssm/processed/"

DATASET_GROUP_DIMS: Dict[Tuple[str, str, str], Tuple[int, int]] = {
    ("perich_miller_population_2018", "c", "center_out_reaching"): (353, 2),
    ("perich_miller_population_2018", "c", "random_target_reaching"): (88, 2),
    ("perich_miller_population_2018", "m", "center_out_reaching"): (159, 2),
    ("perich_miller_population_2018", "m", "random_target_reaching"): (165, 2),
    ("perich_miller_population_2018", "t", "center_out_reaching"): (65, 2),
    ("perich_miller_population_2018", "t", "random_target_reaching"): (73, 2),
    ("perich_miller_population_2018", "j", "center_out_reaching"): (38, 2),
    ("nlb", "j", "center_out_reaching"): (182, 2),
    ("nlb", "j", "maze"): (182, 2),
    ("nlb", "j", "maze_active_target"): (182, 2),
}

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
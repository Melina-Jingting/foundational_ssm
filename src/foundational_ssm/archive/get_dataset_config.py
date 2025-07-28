def get_dataset_config(
    brainset, 
    sessions=None,
    subjects=None,
    exclude_subjects=None, 
    exclude_sessions=None
):
    brainset_norms = {
        "perich_miller_population_2018": {
            "mean": 0.0,
            "std": 20.0
        }
    }

    config = f"""
    - selection:
      - brainset: {brainset}"""

    # Add sessions if provided
    if sessions is not None:
        if not isinstance(sessions, list):
            sessions = [sessions]
        config += "\n        sessions:"
        for session in sessions:
            config += f"\n          - {session}"

    # Add subjects if provided
    if subjects is not None:
        config += "\n        subjects:"
        for subj in subjects:
            config += f"\n          - {subj}"

    # Add exclude clauses if provided
    if exclude_subjects is not None or exclude_sessions is not None:
        config += "\n        exclude:"
        if exclude_subjects is not None:
            if not isinstance(exclude_subjects, list):
                exclude_subjects = [exclude_subjects]
            config += "\n          subjects:"
            for subj in exclude_subjects:
                config += f"\n            - {subj}"
        if exclude_sessions is not None:
            if not isinstance(exclude_sessions, list):
                exclude_sessions = [exclude_sessions]
            config += "\n          sessions:"
            for sess in exclude_sessions:
                config += f"\n            - {sess}"

    config += f"""
      config:
        readout:
          readout_id: cursor_velocity_2d
          normalize_mean: {brainset_norms[brainset]["mean"]}
          normalize_std: {brainset_norms[brainset]["std"]}
          metrics:
            - metric:
                _target_: torchmetrics.R2Score
    """

    config = OmegaConf.create(config)

    return config
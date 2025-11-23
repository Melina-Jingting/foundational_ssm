def generate_predictions_and_activations(model, inputs, state, group_idxs):
    """Generate predictions and activations for a batch"""
    preds, activations_list, state = jax.vmap(
        model.call_with_activations,
        axis_name="batch",
        in_axes=(0, None, 0),
        out_axes=(0, 0, None),
    )(inputs, state, group_idxs)
    activations = {
        f"ssm_block_{i}": activations_list[i] for i in range(len(activations_list))
    }
    return preds, activations


def create_activation_dataset(config):
    """Create a dataset for generating predictions and activations"""
    # Use the same config as validation but with valid_trials split
    activation_dataset = TorchBrainDataset(
        config=config,
        root=DATA_ROOT,
        split="valid_trials",
        transform=transform_brainsets_regular_time_series_smoothed,
    )
    sampling_intervals = activation_dataset.get_sampling_intervals()
    return activation_dataset, sampling_intervals


def generate_session_predictions_and_activations(
    model, state, session_id, trial_intervals, dataset, batch_size=64
):
    """Generate predictions and activations for a single session"""
    logger.info(f"Processing session {session_id}")

    # Collect all trials for this session
    batch = []
    for trial_interval in trial_intervals:
        trial_idx = DatasetIndex(session_id, trial_interval[0], trial_interval[1])
        sample = dataset[trial_idx]
        batch.append(sample)

    batch = pad_collate(batch)
    batch = {k: jax.device_put(np.array(v)) for k, v in batch.items()}

    dataset_group_idxs = batch["dataset_group_idx"]
    inputs = batch["neural_input"]
    targets = batch["behavior_input"]
    mask = batch["mask"]
    num_trials = inputs.shape[0]

    # Split into chunks to avoid memory issues
    preds_list = []
    activations_list = {}

    # Initialize activations dict with empty lists
    num_ssm_blocks = len(model.ssm_blocks)
    for i in range(num_ssm_blocks):
        activations_list[f"ssm_block_{i}"] = []

    for start_idx in tqdm(
        range(0, num_trials, batch_size), desc=f"Processing {session_id}"
    ):
        stop_idx = min(start_idx + batch_size, num_trials)
        preds, activations = generate_predictions_and_activations(
            model,
            inputs[start_idx:stop_idx],
            state,
            dataset_group_idxs[start_idx:stop_idx],
        )
        preds_list.append(preds)
        for ssm_block in activations:
            activations_list[ssm_block].append(activations[ssm_block])

    preds = np.concatenate(preds_list, axis=0)
    activations = {
        key: np.concatenate(value, axis=0) for key, value in activations_list.items()
    }

    return {
        "predictions": preds,
        "activations": activations,
        "targets": targets,
        "mask": mask,
        "num_trials": inputs.shape[0],
    }


def log_predictions_and_activations(model, state, cfg, epoch, current_step, run_name):
    """Generate and log predictions and activations as wandb artifact"""
    print(f"[DEBUG] Generating predictions and activations for epoch {epoch}")

    # Create activation dataset
    activation_dataset, sampling_intervals = create_activation_dataset(
        cfg.activations_dataset_config
    )

    # Generate predictions and activations for each session
    session_predictions_and_activations = {}
    for session_id, trial_intervals in sampling_intervals.items():
        session_data = generate_session_predictions_and_activations(
            model, state, session_id, trial_intervals, activation_dataset
        )
        session_predictions_and_activations[session_id] = session_data

    # Save to H5 file
    h5_path = f"wandb_artifacts/{run_name}/predictions_and_activations.h5"
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        # Create session group
        sessions_group = f.create_group("sessions")

        for session_id, session_data in session_predictions_and_activations.items():
            session_group = sessions_group.create_group(session_id)

            # Save predictions
            session_group.create_dataset(
                "predictions", data=session_data["predictions"]
            )
            session_group.create_dataset("targets", data=session_data["targets"])
            session_group.create_dataset("mask", data=session_data["mask"])
            session_group.create_dataset("num_trials", data=session_data["num_trials"])

            # Save activations
            activations_group = session_group.create_group("activations")
            for block_name, activation_data in session_data["activations"].items():
                activations_group.create_dataset(block_name, data=activation_data)

    # Create and log wandb artifact
    artifact = wandb.Artifact(
        name=f"{run_name}_predictions_and_activation",
        type="predictions_and_activations",
        description=f"Model predictions and activations for epoch {epoch}",
    )
    artifact.add_file(h5_path)

    # Add metadata
    artifact.metadata.update(
        {
            "epoch": epoch,
            "current_step": current_step,
            "num_sessions": len(session_predictions_and_activations),
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        }
    )

    wandb.log_artifact(artifact, aliases=[f"epoch_{epoch}"])
    print(f"[DEBUG] Logged predictions and activations artifact for epoch {epoch}")

    return h5_path

from typing import Any, Dict
import numpy as np
import tensorflow as tf
from foundational_ssm.data_utils.dataset import TorchBrainDataset
from foundational_ssm.data_utils.loaders import DATA_ROOT
from .loaders import _ensure_dim
from foundational_ssm.data_utils import bin_spikes, smooth_spikes
from foundational_ssm.constants import MAX_NEURAL_INPUT_DIM, MAX_BEHAVIOR_INPUT_DIM, DATASET_GROUP_TO_IDX, parse_session_id






# --- OPTIMIZATION 1: Rewrite processing logic using native TensorFlow operations ---


def bin_spikes_tf(spike_timestamps, spike_unit_indices, num_units, sampling_rate, num_bins):
    """
    Bins spike timestamps into a 2D tensor using native TensorFlow operations.
    
    Args:
        spike_timestamps (tf.Tensor): 1D tensor of spike times.
        spike_unit_indices (tf.Tensor): 1D tensor of corresponding unit indices for each spike.
        num_units (int): Total number of units.
        sampling_rate (float): The sampling rate in Hz.
        num_bins (int): The total number of time bins.
        
    Returns:
        tf.Tensor: A 2D tensor of shape [num_units, num_bins].
    """
    bin_indices = tf.cast(tf.floor(spike_timestamps * sampling_rate), dtype=tf.int32)
    
    # Filter out-of-bounds indices to prevent errors
    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    
    # --- FIX: Use tf.where and tf.gather_nd for robust filtering ---
    # This pattern is safer for dynamic shapes than tf.boolean_mask.
    valid_indices = tf.where(valid_mask) # Get indices of True values
    
    # Use the valid indices to gather the elements we want to keep
    spike_unit_indices_filtered = tf.gather_nd(spike_unit_indices, valid_indices)
    bin_indices_filtered = tf.gather_nd(bin_indices, valid_indices)

    # Create indices for scatter_nd of shape [num_valid_spikes, 2]
    scatter_indices = tf.stack([spike_unit_indices_filtered, bin_indices_filtered], axis=1)
    
    # Use tensor_scatter_nd_add to place 1s at the spike locations
    binned_spikes = tf.tensor_scatter_nd_add(
        tensor=tf.zeros((num_units, num_bins), dtype=tf.float32),
        indices=scatter_indices,
        updates=tf.ones(tf.shape(scatter_indices)[0], dtype=tf.float32)
    )
    return binned_spikes

def smooth_spikes_tf(binned_spikes, kern_sd_ms=40.0, bin_size_ms=5.0):
    """
    Applies Gaussian smoothing using a 1D convolution in TensorFlow.
    
    Args:
        binned_spikes (tf.Tensor): The binned spike data of shape [num_units, num_bins].
        kern_sd_ms (float): Standard deviation of Gaussian kernel in ms.
        bin_size_ms (float): Width of time bins in ms.
        
    Returns:
        tf.Tensor: Smoothed spike data.
    """
    # Compute kernel standard deviation in bins
    kern_sd_bins = kern_sd_ms / bin_size_ms
    
    # Create Gaussian window
    win_size = tf.cast(kern_sd_bins * 6, dtype=tf.int32)
    n = tf.range(tf.cast(-win_size // 2, dtype=tf.float32), tf.cast(win_size // 2 + 1, dtype=tf.float32))
    sig2 = 2.0 * kern_sd_bins * kern_sd_bins
    window = tf.exp(-n**2 / sig2)
    window = window / tf.reduce_sum(window)
    
    # Reshape for tf.nn.conv1d. We treat each unit as a channel.
    # Input shape needs to be [batch, in_width, in_channels]
    # Our binned_spikes is [num_units, num_bins], so we transpose and add a batch dim.
    spikes_for_conv = tf.transpose(binned_spikes)[tf.newaxis, :, :]
    
    # Kernel for depthwise conv needs shape [filter_width, in_channels, channel_multiplier]
    # Here, in_channels is num_units, and we want to apply the same filter to each.
    kernel = window[:, tf.newaxis, tf.newaxis] # Shape: [win_size, 1, 1]
    kernel = tf.tile(kernel, [1, tf.shape(spikes_for_conv)[2], 1]) # Shape: [win_size, num_units, 1]

    # Apply convolution along the time axis for each unit independently
    smoothed_data_transposed = tf.nn.depthwise_conv2d(
        spikes_for_conv[:, :, :, tf.newaxis], # Add a height dimension for conv2d
        kernel[:, :, tf.newaxis, :], # Reshape kernel for conv2d
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    
    # Reshape back to [num_units, num_bins]
    return tf.transpose(tf.squeeze(smoothed_data_transposed))


# --- OPTIMIZATION 2: Create a map function that uses these native TF ops ---

def _ensure_dim_tf(tensor, target_dim, axis=1):
    """Native TensorFlow version of the padding/cropping helper function."""
    current_dim = tf.shape(tensor)[axis]
    
    def crop_fn():
        # Use tf.slice for graph-compatible dynamic slicing
        begin = tf.zeros(tf.rank(tensor), dtype=tf.int32)
        size = tf.tensor_scatter_nd_update(
            tf.shape(tensor),
            indices=[[axis]],
            updates=[target_dim]
        )
        return tf.slice(tensor, begin, size)

    def pad_fn():
        # Use tf.tensor_scatter_nd_update for graph-compatible padding setup
        pad_amount = target_dim - current_dim
        paddings = tf.zeros([tf.rank(tensor), 2], dtype=tf.int32)
        paddings = tf.tensor_scatter_nd_update(
            paddings,
            indices=[[axis]],
            updates=[[0, pad_amount]]
        )
        return tf.pad(tensor, paddings, mode="CONSTANT")

    # Use tf.cond to dynamically apply cropping or padding
    return tf.cond(current_dim > target_dim, crop_fn, lambda: tf.cond(current_dim < target_dim, pad_fn, lambda: tensor))



def get_brainset_train_val_loaders_tfds_native(
    train_config=None,
    val_config=None,
    batch_size=32,
    seed=0,
    root=DATA_ROOT,
    num_workers=4,
    keep_files_open=True,
    lazy=True
):
    """
    A fully optimized data loader that uses native TensorFlow operations for processing.
    """
    # --- Shared Dataset Objects ---
    torch_train_dataset = TorchBrainDataset(root=root, config=train_config, split="train", keep_files_open=keep_files_open, lazy=lazy)
    torch_val_dataset = TorchBrainDataset(root=root, config=val_config or train_config, split="valid", keep_files_open=keep_files_open, lazy=lazy)

    # --- Lightweight I/O Function (The only part that uses py_function) ---
    def read_h5_slice(rec_id_tensor, start_tensor, end_tensor, is_train=True):
        rec_id = rec_id_tensor.numpy().decode('utf-8')
        start = start_tensor.numpy()
        end = end_tensor.numpy()
        
        dataset_obj = torch_train_dataset if is_train else torch_val_dataset
        data_sample = dataset_obj.get(rec_id, start, end)
        
        # Perform the simple lookup here in Python
        dataset, subject, task = parse_session_id(data_sample.session.id)
        group_tuple = (dataset, subject, task)
        group_idx = DATASET_GROUP_TO_IDX.get(group_tuple, -1) # Default to -1 if not found

        return (
            data_sample.spikes.timestamps.astype(np.float32),
            data_sample.spikes.unit_index.astype(np.int32),
            data_sample.cursor.vel.astype(np.float32),
            np.array(len(data_sample.units.id), dtype=np.int32),
            np.array(group_idx, dtype=np.int32)
        )

    # --- Native TF Transformation Function ---
    def transform_in_tensorflow(spike_times, spike_indices, cursor_vel, num_units, group_idx):
        # These are now all TensorFlow operations, running inside the graph
        sampling_rate = 100.0
        sampling_window_ms = 1000.0
        num_timesteps = tf.cast(sampling_rate * sampling_window_ms / 1000.0, dtype=tf.int32)

        # 1. Bin and Smooth spikes
        binned = bin_spikes_tf(spike_times, spike_indices, num_units, sampling_rate, num_timesteps)
        smoothed = smooth_spikes_tf(binned)
        neural_input = tf.transpose(smoothed) # Shape: [timesteps, units]
        
        # 2. Pad/crop behavior time dimension
        behavior_input = _ensure_dim_tf(cursor_vel, num_timesteps, axis=0)
        
        # 3. Pad/crop channel dimensions
        neural_input = _ensure_dim_tf(neural_input, MAX_NEURAL_INPUT_DIM, axis=1)
        behavior_input = _ensure_dim_tf(behavior_input, MAX_BEHAVIOR_INPUT_DIM, axis=1)
        
        # # Set shapes explicitly after padding/cropping
        # neural_input.set_shape([num_timesteps, MAX_NEURAL_INPUT_DIM])
        # behavior_input.set_shape([num_timesteps, MAX_BEHAVIOR_INPUT_DIM])
        # group_idx.set_shape([])

        return neural_input, behavior_input, group_idx

    # --- Build the Pipeline ---
    def build_pipeline(dataset_obj, is_train=True):
        sampling_intervals = dataset_obj.get_sampling_intervals()
        window_length = 1.0

        def train_index_generator():
            # Create a flat list of all possible intervals to shuffle them
            all_intervals = []
            for session_name, intervals in sampling_intervals.items():
                for start, end in zip(intervals.start, intervals.end):
                    if end - start >= window_length:
                        all_intervals.append((session_name, start, end))
            
            # Shuffle the intervals at the beginning of each epoch
            np.random.shuffle(all_intervals)

            for session_name, start, end in all_intervals:
                interval_length = end - start
                
                # Apply random offset for jitter, same as in your sampler
                left_offset = np.random.rand() * window_length
                
                current_time = start + left_offset
                last_yielded_time = -1
                while current_time + window_length <= end:
                    yield session_name.encode('utf-8'), np.float32(current_time), np.float32(current_time + window_length)
                    last_yielded_time = current_time
                    current_time += window_length
                
                # Logic to ensure consistent number of samples per interval
                if last_yielded_time > 0:
                    right_offset = end - (last_yielded_time + window_length)
                else:
                    right_offset = end - start - left_offset

                if right_offset + left_offset >= window_length:
                    final_start = end - window_length if right_offset > left_offset else start
                    yield session_name.encode('utf-8'), np.float32(final_start), np.float32(final_start + window_length)
            
        def val_index_generator():
            # Sequential, no random offsets or shuffling for validation
            for session_name, intervals in sampling_intervals.items():
                for start, end in zip(intervals.start, intervals.end):
                    current_time = start
                    while current_time + window_length <= end:
                        yield session_name.encode('utf-8'), np.float32(current_time), np.float32(current_time + window_length)
                        current_time += window_length

        dataset = tf.data.Dataset.from_generator(
            train_index_generator if is_train else val_index_generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            )
        )
        
        if is_train:
            dataset = dataset.shuffle(buffer_size=10000, seed=seed)
        
        dataset = dataset.map(
            lambda rec_id, start, end: tf.py_function(
                func=lambda r, s, e: read_h5_slice(r, s, e, is_train=is_train),
                inp=[rec_id, start, end],
                Tout=[tf.float32, tf.int32, tf.float32, tf.int32, tf.int32] # Updated Tout
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.map(
            transform_in_tensorflow,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    torch_train_dataset.disable_data_leakage_check()
    torch_val_dataset.disable_data_leakage_check()
    
    train_loader = build_pipeline(torch_train_dataset, is_train=True)
    val_loader = build_pipeline(torch_val_dataset, is_train=False)

    return torch_train_dataset, train_loader, torch_val_dataset, val_loader



def transform_to_numpy_dict(
    data: Any,
    *,
    sampling_rate: int = 100,
    sampling_window_ms: int = 1000,
    kern_sd_ms: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Modified transform function that returns NumPy arrays instead of Torch Tensors.
    The internal logic is the same as your original function.
    """
    num_timesteps = int(sampling_rate * sampling_window_ms / 1000)
    bin_size_ms = int(1000 / sampling_rate)
    unit_ids = data.units.id
    # Assuming bin_spikes and smooth_spikes are defined elsewhere
    binned_spikes = bin_spikes(
        spikes=data.spikes,
        num_units=len(unit_ids),
        sampling_rate=sampling_rate,
        num_bins=num_timesteps,
    )
    smoothed_spikes = smooth_spikes(
        binned_spikes,
        kern_sd_ms=kern_sd_ms,
        bin_size_ms=bin_size_ms,
    ).T # Transpose
    behavior_input = data.cursor.vel
    if behavior_input.shape[0] > num_timesteps:
        behavior_input = behavior_input[:num_timesteps]
    elif behavior_input.shape[0] < num_timesteps:
        behavior_input = np.pad(
            behavior_input,
            ((0, num_timesteps - behavior_input.shape[0]), (0, 0)),
            mode="constant",
        )
    dataset, subject, task = parse_session_id(data.session.id)
    group_tuple = (dataset, subject, task)
    group_idx = DATASET_GROUP_TO_IDX[group_tuple]
    smoothed_spikes = _ensure_dim(smoothed_spikes, MAX_NEURAL_INPUT_DIM, axis=1)
    behavior_input = _ensure_dim(behavior_input, MAX_BEHAVIOR_INPUT_DIM, axis=1)

    return {
        "neural_input": smoothed_spikes.astype(np.float32),
        "behavior_input": behavior_input.astype(np.float32),
        "dataset_group_idx": np.array(group_idx, dtype=np.int32),
    }


def get_brainset_train_val_loaders_tfds_optimized(
    recording_id=None,
    train_config=None,
    val_config=None,
    batch_size=32,
    seed=0,
    root=DATA_ROOT,
    transform_fn=transform_to_numpy_dict, # Use the new NumPy-based transform
    num_workers=4, # This will now be used by tf.data.map
    keep_files_open=True, # CRITICAL for performance
    lazy=True, # CRITICAL for performance
):
    """
    An optimized version of the data loader that leverages tf.data.map for parallelism.
    """
    # --- Train Dataset ---
    torch_train_dataset = TorchBrainDataset(
        root=root,
        recording_id=recording_id,
        config=train_config,
        split="train",
        keep_files_open=keep_files_open,
        lazy=lazy,
    )
    torch_train_dataset.disable_data_leakage_check()
    
    train_sampling_intervals = torch_train_dataset.get_sampling_intervals()
    window_length = 1.0 # Assuming this is fixed based on your sampler

    # --- OPTIMIZATION: Create a streaming generator for INDICES only ---
    # This generator implements the logic from RandomFixedWindowSampler directly
    # but yields indices immediately, avoiding high memory usage.
    def train_index_generator():
        # Create a flat list of all possible intervals to shuffle them
        all_intervals = []
        for session_name, intervals in train_sampling_intervals.items():
            for start, end in zip(intervals.start, intervals.end):
                if end - start >= window_length:
                    all_intervals.append((session_name, start, end))
        
        # Shuffle the intervals at the beginning of each epoch
        np.random.shuffle(all_intervals)

        for session_name, start, end in all_intervals:
            interval_length = end - start
            
            # Apply random offset for jitter, same as in your sampler
            left_offset = np.random.rand() * window_length
            
            current_time = start + left_offset
            last_yielded_time = -1
            while current_time + window_length <= end:
                yield session_name.encode('utf-8'), np.float32(current_time), np.float32(current_time + window_length)
                last_yielded_time = current_time
                current_time += window_length
            
            # Logic to ensure consistent number of samples per interval
            if last_yielded_time > 0:
                right_offset = end - (last_yielded_time + window_length)
            else:
                right_offset = end - start - left_offset

            if right_offset + left_offset >= window_length:
                final_start = end - window_length if right_offset > left_offset else start
                yield session_name.encode('utf-8'), np.float32(final_start), np.float32(final_start + window_length)

    # This function will be executed in parallel by TensorFlow's backend.
    def load_and_transform_sample(rec_id_tensor, start_tensor, end_tensor, dataset_obj):
        rec_id = rec_id_tensor.numpy().decode('utf-8')
        start = start_tensor.numpy()
        end = end_tensor.numpy()
        
        data_sample = dataset_obj.get(rec_id, start, end)
        
        processed_sample = transform_fn(data_sample)

        return (
            processed_sample["neural_input"],
            processed_sample["behavior_input"],
            processed_sample["dataset_group_idx"],
        )

    tf_train_dataset = tf.data.Dataset.from_generator(
        train_index_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )
    tf_train_dataset = tf_train_dataset.shuffle(buffer_size=10000, seed=seed)
    tf_train_dataset = tf_train_dataset.map(
        lambda rec_id, start, end: tf.py_function(
            func=lambda r, s, e: load_and_transform_sample(r, s, e, dataset_obj=torch_train_dataset),
            inp=[rec_id, start, end],
            Tout=[tf.float32, tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    tf_train_dataset = tf_train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    torch_val_dataset = TorchBrainDataset(
        root=root,
        recording_id=recording_id,
        config=val_config or train_config,
        split="valid",
        keep_files_open=keep_files_open,
        lazy=lazy,
    )
    val_sampling_intervals = torch_val_dataset.get_sampling_intervals()

    def val_index_generator():
        # Sequential, no random offsets or shuffling for validation
        for session_name, intervals in val_sampling_intervals.items():
            for start, end in zip(intervals.start, intervals.end):
                current_time = start
                while current_time + window_length <= end:
                    yield session_name.encode('utf-8'), np.float32(current_time), np.float32(current_time + window_length)
                    current_time += window_length

    tf_val_dataset = tf.data.Dataset.from_generator(
        val_index_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )
    )
    
    tf_val_dataset = tf_val_dataset.map(
        lambda rec_id, start, end: tf.py_function(
            # Re-use the same processing function, but with the validation dataset object
            func=lambda r, s, e: load_and_transform_sample(r, s, e, dataset_obj=torch_val_dataset),
            inp=[rec_id, start, end],
            Tout=[tf.float32, tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    torch_val_dataset.disable_data_leakage_check()
    
    tf_val_dataset = tf_val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return torch_train_dataset, tf_train_dataset, torch_val_dataset, tf_val_dataset

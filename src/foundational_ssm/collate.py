import torch



def pad_collate(batch, fixed_seq_len=None):
    # Assume batch is a list of dicts with keys: 'neural_input', 'behavior_input', etc.
    # Each 'neural_input' is a tensor of shape (timesteps, units)
    neural_inputs = [item['neural_input'].squeeze(0) for item in batch if item is not None]  # (timesteps, units)
    behavioral_inputs = [item['behavior_input'].squeeze(0) for item in batch if item is not None]
    
    # Determine the fixed sequence length
    if fixed_seq_len is None:
        max_len = max(x.shape[0] for x in neural_inputs)
    else:
        max_len = fixed_seq_len

    # Pad or truncate each sequence to fixed length
    def pad_or_truncate(tensor, max_len):
        seq_len = tensor.shape[0]
        if seq_len == max_len:
            return tensor
        elif seq_len > max_len:
            return tensor[:max_len]
        else:
            pad_shape = (max_len - seq_len,) + tensor.shape[1:]
            pad_tensor = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad_tensor], dim=0)

    padded_neural = torch.stack([pad_or_truncate(x, max_len) for x in neural_inputs if x is not None])  # (batch, max_len, units)
    padded_behavior = torch.stack([pad_or_truncate(x, max_len) for x in behavioral_inputs if x is not None])

    # Create mask: 1 for real data, 0 for padding
    lengths = [x.shape[0] for x in neural_inputs]
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :min(l, max_len)] = 1

    # Stack other fields (e.g., dataset_group_idx)
    dataset_group_idx = torch.stack([item['dataset_group_idx'] for item in batch])
    session_date = torch.stack([item['session_date'] for item in batch])

    return {
        'neural_input': padded_neural,
        'behavior_input': padded_behavior,
        'mask': mask,
        'dataset_group_idx': dataset_group_idx,
        'session_date': session_date,
        # add other fields as needed
    }
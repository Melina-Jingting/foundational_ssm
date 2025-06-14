import os
import sys
import warnings
import logging
import torch

# Suppress warnings and logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

# Core imports
from config import get_default_config, device, parse_args_and_load_config, print_config_summary
from models import SSMNeuroModel
from losses import CombinedLoss
from foundational_ssm.trainer import train
from plotting import plot_training_curves
from foundational_ssm.metrics import ValidationMetrics
from utils import get_train_val_loaders, get_dataset_config

def custom_collate(batch):
    """
    Custom collate function to handle variable-length sequences.
    This function will pad or trim sequences to ensure they have consistent length.
    """
    # Extract batch tensors
    neural_inputs = [item['neural_input'] for item in batch]
    behavior_inputs = [item['behavior_input'] for item in batch]
    
    # Get tensor shapes
    neural_shapes = [x.shape[0] for x in neural_inputs]
    behavior_shapes = [x.shape[0] for x in behavior_inputs]
    
    # Determine common length (the minimum for trimming)
    min_length = min(min(neural_shapes), min(behavior_shapes))
    
    # Trim all sequences to the minimum length
    trimmed_batch = []
    for item in batch:
        trimmed_item = {
            'neural_input': item['neural_input'][:min_length],
            'behavior_input': item['behavior_input'][:min_length],
            'session_id': item['session_id'],
            'subject_id': item['subject_id']
        }
        
        if 'neural_target' in item:
            trimmed_item['neural_target'] = item['neural_target'][:min_length]
        
        trimmed_batch.append(trimmed_item)
    
    # Stack trimmed tensors
    final_batch = {
        'neural_input': torch.stack([item['neural_input'] for item in trimmed_batch]),
        'behavior_input': torch.stack([item['behavior_input'] for item in trimmed_batch]),
        'session_id': [item['session_id'] for item in trimmed_batch],
        'subject_id': [item['subject_id'] for item in trimmed_batch]
    }
    
    if 'neural_target' in trimmed_batch[0]:
        final_batch['neural_target'] = torch.stack([item['neural_target'] for item in trimmed_batch])
    
    return final_batch

def main():
    # Get configuration
    config, args = parse_args_and_load_config()
    print(f"Running with device: {device}")
    
    # Load dataset
    train_dataset, train_loader, val_dataset, val_loader = get_train_val_loaders(
        train_config=get_dataset_config(
            config.dataset.name,
            subjects=config.dataset.subjects
        ),
        batch_size=config.dataset.batch_size
    )
    
    # Print dataset info
    num_units = len(train_dataset.get_unit_ids())
    print(f"Num Units in Session: {num_units}")

    # Initialize model
    model = SSMNeuroModel(
        num_neural_features=config.model.num_neural_features,
        num_behavior_features=config.model.num_behavior_features,
        num_context_features=config.model.num_context_features,
        embedding_dim=config.model.embedding_dim,
        ssm_projection_dim=config.model.ssm_projection_dim,
        ssm_hidden_dim=config.model.ssm_hidden_dim,
        ssm_num_layers=config.model.ssm_num_layers,
        ssm_dropout=config.model.ssm_dropout,
        pred_neural_dim=config.model.pred_neural_dim,
        pred_behavior_dim=config.model.pred_behavior_dim,
        sequence_length=config.model.sequence_length,
        sampling_rate=config.model.sampling_rate,
        lin_dropout=config.model.lin_dropout,
        activation_fn=config.model.activation_fn,
        subject_ids=train_dataset.get_subject_ids()
    )
    model = model.to(device)

    # Initialize vocabularies
    model.session_emb.initialize_vocab(train_dataset.get_session_ids())
    model.unit_emb.initialize_vocab(train_dataset.get_unit_ids())

    # Connect tokenizer to datasets
    transform = model.tokenize
    train_dataset.transform = transform
    val_dataset.transform = transform

    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    loss_fn = CombinedLoss(
        neural_weight=config.training.neural_weight,
        behavior_weight=config.training.behavior_weight
    )

    # Optional: Test validation metrics on a small batch before full training
    if config.get('test_metrics_before_training', False):
        validator = ValidationMetrics(device)
        validator.test_on_small_batch(model, val_dataset, val_loader)

    # Train the model
    train_losses, val_metrics_history = train(
        model, 
        optimizer, 
        train_loader, 
        val_loader, 
        loss_fn, 
        config
    )

    
    
    # Save the model
    os.makedirs("models/saved", exist_ok=True)
    model_path = f"models/saved/{config.wandb.run_name}_{args.config}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, train_losses, val_metrics_history

if __name__ == "__main__":
    main()
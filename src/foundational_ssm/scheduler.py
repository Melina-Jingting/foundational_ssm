import optax

def create_cosine_annealing_scheduler(initial_lr, total_steps, min_lr=0.0, warmup_steps=0):
    """
    Creates a cosine annealing learning rate scheduler with optional warmup.
    
    Args:
        initial_lr: Initial learning rate
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0.0)
        warmup_steps: Number of warmup steps (default: 0)
        
    Returns:
        optax scheduler function
    """
    if warmup_steps > 0:
        # Warmup followed by cosine annealing
        warmup_scheduler = optax.linear_schedule(
            init_value=0.0,
            end_value=initial_lr,
            transition_steps=warmup_steps
        )
        
        cosine_scheduler = optax.cosine_decay_schedule(
            init_value=initial_lr,
            decay_steps=total_steps - warmup_steps,
            alpha=min_lr / initial_lr  # alpha determines the minimum value
        )
        
        # Combine warmup and cosine annealing
        scheduler = optax.join_schedules(
            schedules=[warmup_scheduler, cosine_scheduler],
            boundaries=[warmup_steps]
        )
    else:
        # Pure cosine annealing
        scheduler = optax.cosine_decay_schedule(
            init_value=initial_lr,
            decay_steps=total_steps,
            alpha=min_lr / initial_lr
        )
    
    return scheduler
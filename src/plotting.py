import matplotlib.pyplot as plt
import torch 


def print_model(model: torch.nn.Module):
    """
    Prints a summary of the model architecture and parameter count.
    """
    model_str = str(model).split('\n')
    print("\nModel:")
    print('\n'.join(model_str[:5]))
    print("...")
    print('\n'.join(model_str[-min(5, len(model_str)):]))
    num_params = sum(p.numel() for p in model.parameters())
    if num_params > 1e9:
        param_str = f"{num_params/1e9:.1f}G"
    elif num_params > 1e6:
        param_str = f"{num_params/1e6:.1f}M"
    else:
        param_str = f"{num_params/1e3:.1f}K"
    print(f"\nNumber of parameters: {param_str}\n")


def plot_training_curves(r2_log, loss_log):
    """
    Plots the training curves: training loss and validation R2 score.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, len(loss_log), len(loss_log)), loss_log)
    plt.title("Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(r2_log)
    plt.title("Validation R2")
    plt.xlabel("Epochs")
    plt.ylabel("R2 Score")
    plt.grid()
    plt.tight_layout()
    plt.show()
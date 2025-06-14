import torch
import torch.nn.functional as F
from torch import nn
from src.utils import move_to_gpu
from typing import Dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class CombinedLoss(nn.Module):
    def __init__(self, neural_weight=1.0, behavior_weight=1.0):
        super().__init__()
        self.neural_weight = neural_weight
        self.behavior_weight = behavior_weight

    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        loss_neural = F.poisson_nll_loss(
            input=predictions["pred_neural"],
            target=targets["neural_input"],
            log_input=False,
            full=True,
            reduction='mean'
        )

        # Behavior Loss: MSE
        loss_behavior = F.mse_loss(
            input=predictions["pred_behavior"],
            target=targets["behavior_input"],
            reduction='mean'
        )
        
        # print(f"Loss Neural: {loss_neural.item()}, Loss Behavior: {loss_behavior.item()}")
        combined_loss = (self.neural_weight * loss_neural) + \
                        (self.behavior_weight * loss_behavior)
        return combined_loss
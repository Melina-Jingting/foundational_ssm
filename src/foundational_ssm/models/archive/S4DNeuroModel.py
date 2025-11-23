# Import model components
from .s4d import S4D
import torch.nn as nn


class S4DNeuroModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_state=64,
        num_layers=2,
        hidden_dim=64,
        dropout=0.1,
        ssm_core: str = "s4d",
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Stack of S4D layers
        self.ssm_block = nn.Sequential(
            *[
                S4D(hidden_dim, d_state=d_state, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, neural_input, behavior_input=None, session_id=None, subject_id=None
    ):
        x = neural_input.transpose(1, 2)

        x = self.input_projection(x.transpose(1, 2)).transpose(1, 2)

        x = self.ssm_block(x)

        # Final projection [batch, hidden, time] -> [batch, time, output_channels]
        x = x.transpose(1, 2)
        x = self.output_projection(x)

        return x

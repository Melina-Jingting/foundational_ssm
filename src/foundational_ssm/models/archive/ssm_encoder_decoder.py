import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from temporaldata import Data
from foundational_ssm.data_utils import bin_spikes, map_binned_features_to_global
from foundational_ssm.models.foundational import InfiniteVocabEmbedding, LRU


class SMED(nn.Module):
    """
    SSM-based model for neural decoding, inspired by the provided diagram.
    It processes multiple input modalities (Neural, Behavior, Stimuli, Context),
    embeds them, passes them through an SSM core, and then decodes them into
    multiple output predictions.
    """

    def __init__(
        self,
        # Input feature dimensions for each modality (before embedding)
        num_neural_features: int,
        num_behavior_features: int,
        num_context_features: int,
        # Embedding dimension (D in the diagram)
        subject_ids: List[
            str
        ],  # List of subject IDs for which the model is initialized
        embedding_dim: int,
        # SSM core dimensions
        ssm_projection_dim: int,  # Dimension after initial projection, M in diagram
        ssm_hidden_dim: int,  # Hidden dimension of SSM blocks
        ssm_num_layers: int,
        ssm_dropout: float,
        # Output feature dimensions for each prediction head
        pred_neural_dim: int,
        pred_behavior_dim: int,
        # General params
        sequence_length: float,  # Max duration of input sequence (seconds)
        sampling_rate: float = 100,  # Hz, e.g., for converting sec to samples
        lin_dropout: float = 0.1,
        activation_fn: str = "relu",  # Type of activation: "relu", "gelu", "tanh", etc.
        embed_init_scale: float = 0.02,
        bin_size: float = 1e-3,  # Size of bins for neural features (seconds)
    ):
        super().__init__()

        self.num_neural_features = num_neural_features
        self.num_behavior_features = num_behavior_features
        self.num_context_features = num_context_features
        self.embedding_dim = embedding_dim
        self.ssm_projection_dim = ssm_projection_dim
        self.ssm_hidden_dim = ssm_hidden_dim
        self.sequence_length_sec = sequence_length
        self.sampling_rate = sampling_rate
        # Calculate sequence length in samples (timesteps)
        self.num_timesteps = int(sequence_length * sampling_rate)
        self.lin_dropout_rate = lin_dropout
        self.embed_init_scale = embed_init_scale
        self.subject_ids = subject_ids

        # Activation function
        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        # 1. Tokenization Embeddings
        self.session_emb = InfiniteVocabEmbedding(
            embedding_dim=self.num_context_features, init_scale=self.embed_init_scale
        )

        self.unit_emb = InfiniteVocabEmbedding(
            embedding_dim=self.embedding_dim, init_scale=self.embed_init_scale
        )

        # 2. Subject-Specific Embedders (map modality features to embedding_dim)
        self.neural_embedders = nn.ModuleDict(
            {
                subj_id: nn.Linear(self.num_neural_features, self.embedding_dim)
                for subj_id in self.subject_ids
            }
        )
        self.behavior_embedders = nn.ModuleDict(
            {
                subj_id: nn.Linear(self.num_behavior_features, self.embedding_dim)
                for subj_id in self.subject_ids
            }
        )

        self.context_embedder = nn.Linear(self.num_context_features, self.embedding_dim)

        # Total dimension after concatenating D-dimensional embeddings from 4 modalities
        num_active_modalities = (
            3  # Neural, Behavior, Context (No Stimuli in this version)
        )
        concatenated_dim = self.embedding_dim * num_active_modalities

        # 3. Foundational SSM Core
        # self.projection_to_ssm_input = nn.Linear(concatenated_dim, ssm_projection_dim)
        self.ssm_blocks = nn.ModuleList()
        for i in range(ssm_num_layers):
            self.ssm_blocks.append(
                LRU(
                    in_features=concatenated_dim,
                    out_features=concatenated_dim,
                    state_features=ssm_hidden_dim,
                )
            )
        self.ssm_output_dim = concatenated_dim

        # 4. Subject-Specific Decoders
        self.decoder_neural_modules = nn.ModuleDict(
            {
                subj_id: nn.Linear(self.ssm_output_dim, pred_neural_dim)
                for subj_id in self.subject_ids
            }
        )
        self.decoder_behavior_modules = nn.ModuleDict(
            {
                subj_id: nn.Linear(self.ssm_output_dim, pred_behavior_dim)
                for subj_id in self.subject_ids
            }
        )

        self.dropout = nn.Dropout(self.lin_dropout_rate)

    def forward(
        self,
        neural_input: torch.Tensor,  # Shape: (batch, seq_len, num_neural_features)
        behavior_input: torch.Tensor,  # Shape: (batch, seq_len, num_behavior_features)
        session_id: torch.Tensor,  # Shape: (batch)
        subject_id: List[str],  # Shape: (batch)
        neural_mask: Optional[torch.Tensor] = None,
        behavior_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.unit_emb.is_lazy():
            raise ValueError(
                "Unit vocabulary has not been initialized, please use "
                "`model.unit_emb.initialize_vocab(unit_ids)`"
            )
        if self.session_emb.is_lazy():
            raise ValueError(
                "Session vocabulary has not been initialized, please use "
                "`model.session_emb.initialize_vocab(session_ids)`"
            )

        batch_size, seq_len, _ = neural_input.shape
        device = neural_input.device

        session_tokens = [self.session_emb.tokenizer(sid) for sid in session_id]
        session_embs = torch.stack(
            [
                self.session_emb(
                    torch.tensor(tokens, device=device).unsqueeze(0)
                ).squeeze(0)
                for tokens in session_tokens
            ]
        )
        context_input = session_embs.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # Shape: (batch, seq, embed_dim)

        embedded_neural = []
        embedded_behavior = []

        # Apply masks if provided
        if neural_mask is not None:
            neural_mask = neural_mask.to(device)
            neural_input = neural_input * neural_mask.unsqueeze(1).unsqueeze(2)
        if behavior_mask is not None:
            behavior_mask = behavior_mask.to(device)
            behavior_input = behavior_input * behavior_mask.unsqueeze(1).unsqueeze(2)

        # 1. Embed inputs
        for i in range(batch_size):
            subj_id = subject_id[i]
            if subj_id not in self.subject_ids:
                raise ValueError(
                    f"Unknown subject_id '{subj_id}' in batch. Model not initialized for this subject."
                )

            emb_n = self.neural_embedders[subj_id](
                neural_input[i]
            )  # Shape: (seq, num_neural_features)
            embedded_neural.append(self.dropout(self.activation(emb_n)))

            emb_b = self.behavior_embedders[subj_id](
                behavior_input[i]
            )  # Shape: (seq, num_behavior_features)
            embedded_behavior.append(self.dropout(self.activation(emb_b)))

        embedded_neural = torch.stack(
            embedded_neural, dim=0
        )  # Shape: (batch, seq_len, embedding_dim)
        embedded_behavior = torch.stack(
            embedded_behavior, dim=0
        )  # Shape: (batch, seq_len, embedding_dim)
        embedded_context = self.dropout(
            self.activation(self.context_embedder(context_input))
        )  # Shape: (batch, seq_len, embedding_dim)

        # 2. Concatenate embeddings
        # Shape: (batch, seq_len, embedding_dim * 3)
        concatenated_embeddings = torch.cat(
            [embedded_neural, embedded_behavior, embedded_context], dim=-1
        )

        # 3. Pass through Foundational SSM Core
        # ssm_core_input = self.dropout(self.activation(self.projection_to_ssm_input(concatenated_embeddings)))
        ssm_core_input = (
            concatenated_embeddings  # Shape: (batch, seq_len, concatenated_dim)
        )
        ssm_layer_output = ssm_core_input
        for ssm_block in self.ssm_blocks:
            ssm_layer_output = ssm_block(
                ssm_layer_output
            )  # Adjust if it returns state tuple
            if isinstance(ssm_layer_output, tuple):  # e.g. LSTM output, hidden
                ssm_layer_output = ssm_layer_output[0]
            ssm_layer_output = self.dropout(
                ssm_layer_output
            )  # General dropout after each block processing
        final_ssm_output = (
            ssm_layer_output  # Shape: (batch, seq_len, self.ssm_output_dim)
        )

        # 4. Decode
        all_pred_neural = []
        all_pred_behavior = []

        for i in range(batch_size):
            subj_id = subject_id[i]
            ssm_out_sample = final_ssm_output[i]  # (seq_len, ssm_output_dim)

            raw_pred_n = self.decoder_neural_modules[subj_id](ssm_out_sample)
            pred_n = F.softplus(
                raw_pred_n
            )  # Ensure positive predictions for firing rates
            all_pred_neural.append(pred_n)

            pred_b = self.decoder_behavior_modules[subj_id](ssm_out_sample)
            all_pred_behavior.append(pred_b)

        predictions = {
            "pred_neural": torch.stack(all_pred_neural),
            "pred_behavior": torch.stack(all_pred_behavior),
        }

        return predictions

    def tokenize(self, data: Data) -> Dict:
        r"""Tokenizer used to tokenize Data for the POYO model.

        This tokenizer can be called as a transform. If you are applying multiple
        transforms, make sure to apply this one last.

        This code runs on CPU. Do not access GPU tensors inside this function.
        """

        unit_ids = data.units.id
        spikes = data.spikes
        binned_spikes = bin_spikes(
            spikes=spikes,
            num_units=len(unit_ids),
            bin_size=1 / self.sampling_rate,
            num_bins=self.num_timesteps,
        ).T
        neural_input = map_binned_features_to_global(
            session_binned_features=binned_spikes,
            session_unit_id_strings=unit_ids,
            max_global_units=self.num_neural_features,
        )  # (N_timesteps, N_global_units)

        behavior_input = data.cursor.vel  # (N_timesteps, N_behavior_features)

        data_dict = {
            "neural_input": torch.tensor(neural_input, dtype=torch.float32),
            "behavior_input": torch.tensor(behavior_input, dtype=torch.float32),
            "session_id": data.session.id,
            "subject_id": data.subject.id,
        }

        return data_dict

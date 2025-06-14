import torch
import torch.nn.functional as F
from utils import move_to_gpu



def r2_score(y_pred, y_true):
    # Compute total sum of squares (variance of the true values)
    y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)

    # Compute residual sum of squares
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # Compute R^2
    r2 = 1 - ss_res / ss_total

    return r2

class ValidationMetrics:
    def __init__(self, device):
        self.device = device
        
    def compute_metrics(self, dataloader, model):
        model.eval()
        
        # Initialize overall metrics
        metrics = {
            "encoding_loss": 0.0,  
            "decoding_loss": 0.0,  
            "combined_loss": 0.0,
            "behavior_r2": 0.0
        }
        
        # Initialize per-subject metrics
        subject_metrics = {}
        for subj_id in model.subject_ids:
            subject_metrics[subj_id] = {
                "encoding_loss": 0.0,
                "decoding_loss": 0.0,
                "combined_loss": 0.0,
                "behavior_r2": 0.0,
                "sample_count": 0
            }
        
        # Collect predictions for R² calculation
        all_behavior_targets = []
        all_behavior_preds = []
        
        # Per-subject predictions and targets
        subject_behavior_targets = {subj_id: [] for subj_id in model.subject_ids}
        subject_behavior_preds = {subj_id: [] for subj_id in model.subject_ids}
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = move_to_gpu(batch, self.device)
                batch_size = batch["behavior_input"].shape[0]
                
                # Forward pass for encoding (neural prediction)
                encoding_predictions = model(
                    **batch,
                    neural_mask=torch.zeros(batch_size, device=self.device),
                    behavior_mask=torch.ones(batch_size, device=self.device)
                )
                
                # Forward pass for decoding (behavior prediction)
                decoding_predictions = model(
                    **batch,
                    neural_mask=torch.ones(batch_size, device=self.device),
                    behavior_mask=torch.zeros(batch_size, device=self.device)
                )
                
                # Overall metrics
                encoding_loss = F.poisson_nll_loss(
                    input=encoding_predictions["pred_neural"],
                    target=batch["neural_input"],
                    log_input=False,
                    full=True,
                    reduction='mean'
                )
                
                decoding_loss = F.mse_loss(
                    input=decoding_predictions["pred_behavior"],
                    target=batch["behavior_input"],
                    reduction='mean'
                )
                
                combined_loss = encoding_loss + decoding_loss
                
                metrics["encoding_loss"] += encoding_loss.item()
                metrics["decoding_loss"] += decoding_loss.item()
                metrics["combined_loss"] += combined_loss.item()
                
                # Collect overall predictions
                all_behavior_targets.append(batch["behavior_input"])
                all_behavior_preds.append(decoding_predictions["pred_behavior"])
                
                # Per-subject metrics
                for i, subj_id in enumerate(batch["subject_id"]):
                    # Calculate per-subject losses
                    subj_encoding_loss = F.poisson_nll_loss(
                        input=encoding_predictions["pred_neural"][i:i+1],
                        target=batch["neural_input"][i:i+1],
                        log_input=False,
                        reduction='mean'
                    )
                    
                    subj_decoding_loss = F.mse_loss(
                        input=decoding_predictions["pred_behavior"][i:i+1],
                        target=batch["behavior_input"][i:i+1],
                        reduction='mean'
                    )
                    
                    subject_metrics[subj_id]["encoding_loss"] += subj_encoding_loss.item()
                    subject_metrics[subj_id]["decoding_loss"] += subj_decoding_loss.item()
                    subject_metrics[subj_id]["combined_loss"] += (subj_encoding_loss + subj_decoding_loss).item()
                    subject_metrics[subj_id]["sample_count"] += 1
                    
                    # Store per-subject predictions for R2 calculation
                    subject_behavior_targets[subj_id].append(batch["behavior_input"][i:i+1])
                    subject_behavior_preds[subj_id].append(decoding_predictions["pred_behavior"][i:i+1])
                
                num_batches += 1
        
        # Average overall metrics
        for key in ["encoding_loss", "decoding_loss", "combined_loss"]:
            metrics[key] /= num_batches if num_batches > 0 else 1
            
        # Calculate overall behavior R²
        behavior_targets = torch.cat(all_behavior_targets)
        behavior_preds = torch.cat(all_behavior_preds)
        metrics["behavior_r2"] = r2_score(behavior_preds.cpu(), behavior_targets.cpu())
        
        # Calculate per-subject R² and average per-subject metrics
        for subj_id in model.subject_ids:
            if subject_metrics[subj_id]["sample_count"] > 0:
                # Average per-subject losses
                count = subject_metrics[subj_id]["sample_count"]
                for key in ["encoding_loss", "decoding_loss", "combined_loss"]:
                    subject_metrics[subj_id][key] /= count
                
                # Calculate per-subject R²
                if len(subject_behavior_targets[subj_id]) > 0:
                    subj_targets = torch.cat(subject_behavior_targets[subj_id])
                    subj_preds = torch.cat(subject_behavior_preds[subj_id])
                    subject_metrics[subj_id]["behavior_r2"] = r2_score(
                        subj_preds.cpu(), subj_targets.cpu()
                    )
        
        # Add per-subject metrics to the overall metrics
        metrics["per_subject"] = subject_metrics
        
        return metrics
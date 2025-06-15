import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import ctypes
import json
from datetime import datetime
import argparse

# Constants - keeping same as part1 for consistency
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "open_llama_3b"))
BATCH_SIZE = 2048
NUM_TOKENS = 5_000_000  # Same as part1
LEARNING_RATE = 1e-5  
NUM_FEATURES = 50_000  # Same feature size as part1
L1_LAMBDA = 1.0  # L1 regularization strength
TARGET_LAYER = 16  # Same target layer as part1
TRAIN_STEPS = 200_000  # Same number of training steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints_residual"  # Different checkpoint directory
CHECKPOINT_INTERVAL = 10_000
VALIDATION_INTERVAL = 1000
VALIDATION_SPLIT = 0.1

# Reuse utility functions from part1
def check_tensor(tensor, name="tensor", print_stats=True):
    """Check tensor for NaN, Inf, and basic statistics."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf or print_stats:
        print(f"\n--- Checking {name} ---")
        print(f"Has NaN: {has_nan}")
        print(f"Has Inf: {has_inf}")
        
        if print_stats:
            try:
                print(f"Shape: {tensor.shape}")
                print(f"Min: {tensor.min().item()}")
                print(f"Max: {tensor.max().item()}")
                print(f"Mean: {tensor.mean().item()}")
                print(f"Std: {tensor.std().item()}")
            except RuntimeError as e:
                print(f"Could not compute stats: {e}")
    
    return has_nan or has_inf

def debug_model_parameters(model, name="model"):
    """Check model parameters for NaN and Inf values."""
    print(f"\n--- Checking {name} parameters ---")
    for param_name, param in model.named_parameters():
        has_issue = check_tensor(param, f"{param_name}", print_stats=False)
        if has_issue:
            print(f"Issue detected in {param_name}")
            check_tensor(param, f"{param_name}", print_stats=True)

def prevent_sleep():
    """Prevent Windows from sleeping during training."""
    ctypes.windll.kernel32.SetThreadExecutionState(
        0x80000002  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )

def allow_sleep():
    """Allow Windows to sleep again."""
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS

def save_checkpoint(model, optimizer, scheduler, step, best_loss, metrics_history, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_loss': best_loss,
        'metrics_history': metrics_history,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save metrics separately for easy access
    metrics_path = checkpoint_dir / f"metrics_step_{step}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    # Save best model if this is the best loss
    if metrics_history['val_best_loss'] == best_loss:
        best_model_path = checkpoint_dir / "best_residual_model.pt"
        torch.save(model.state_dict(), best_model_path)

def compute_l0_sparsity(features, threshold=1e-6):
    """Compute L0 'norm' (count of non-zero elements) for features."""
    with torch.no_grad():
        zeros = (torch.abs(features) < threshold).float().mean().item()
        return zeros

# SAE class - same architecture as part1
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Use same initialization as part1
        nn.init.normal_(self.decoder.weight, std=0.0001)
        self.encoder[0].weight.data = self.decoder.weight.data.T
        nn.init.zeros_(self.encoder[0].bias)
        
        print("\n--- Checking initialization ---")
        check_tensor(self.encoder[0].weight, "encoder.weight", True)
        check_tensor(self.encoder[0].bias, "encoder.bias", True)
        check_tensor(self.decoder.weight, "decoder.weight", True)
    
    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            check_tensor(x, "forward_input")
            
        features = self.encoder(x)
        
        if torch.isnan(features).any() or torch.isinf(features).any():
            check_tensor(features, "features")
            
        reconstruction = self.decoder(features)
        
        if torch.isnan(reconstruction).any() or torch.isinf(reconstruction).any():
            check_tensor(reconstruction, "reconstruction")
            
        return features, reconstruction

class ResidualDataset(Dataset):
    """Dataset that holds original activations and residual errors."""
    def __init__(self, original_activations, residual_errors):
        self.original_activations = original_activations
        self.residual_errors = residual_errors
        assert len(original_activations) == len(residual_errors)
    
    def __len__(self):
        return len(self.residual_errors)
    
    def __getitem__(self, idx):
        return {
            'original': self.original_activations[idx],
            'residual': self.residual_errors[idx]
        }

def collect_activations(model, tokenizer, num_tokens):
    """Collect activations from the model's target layer - same as part1."""
    activations = []
    
    # Same target indices as part1
    target_indices = set()
    target_chunks = [
        (14000, 15000), (16000, 17000), (66000, 67000), (111000, 112000), (147000, 148000),
        (165000, 166000), (182000, 183000), (187000, 188000), (251000, 252000), (290000, 291000),
        (295000, 296000), (300000, 301000), (313000, 314000), (343000, 344000), (366000, 367000),
        (367000, 368000), (380000, 381000), (400000, 401000), (407000, 408000), (420000, 421000),
        (440000, 441000), (443000, 444000), (479000, 480000), (480000, 481000), (514000, 515000),
        (523000, 524000), (552000, 553000), (579000, 580000), (583000, 584000), (616000, 617000),
        (659000, 660000), (663000, 664000), (690000, 691000), (810000, 811000), (824000, 825000),
        (876000, 877000), (881000, 882000), (908000, 909000), (969000, 970000), (970000, 971000),
        (984000, 985000), (990000, 991000), (995000, 996000), (997000, 998000), (1000000, 1001000),
        (1024000, 1025000), (1099000, 1100000), (1127000, 1128000), (1163000, 1164000), (1182000, 1183000),
        (1209000, 1210000), (1253000, 1254000), (1266000, 1267000), (1270000, 1271000), (1276000, 1277000),
        (1290000, 1291000), (1307000, 1308000), (1326000, 1327000), (1345000, 1346000), (1359000, 1360000),
        (1364000, 1365000), (1367000, 1368000), (1385000, 1386000), (1391000, 1392000), (1468000, 1469000),
        (1508000, 1509000), (1523000, 1524000), (1539000, 1540000), (1574000, 1575000), (1583000, 1584000),
        (1590000, 1591000), (1593000, 1594000), (1599000, 1600000), (1627000, 1628000), (1679000, 1680000),
        (1690000, 1691000), (1691000, 1692000), (1782000, 1783000), (1788000, 1789000)
    ]
    
    for start, end in target_chunks:
        target_indices.update(range(start, end))
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    
    batch_count = 0
    for idx, sample in enumerate(tqdm(dataset, desc="Collecting activations")):
        if idx not in target_indices:
            continue
            
        inputs = tokenizer(sample["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[TARGET_LAYER]
            
            if check_tensor(hidden_states, "hidden_states_batch", print_stats=False):
                print(f"Found problematic hidden states in batch")
                
            batch_size, seq_len, hidden_size = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_size)
            activations.append(hidden_states.cpu())
        
        batch_count += 1
        if batch_count >= num_tokens // BATCH_SIZE:
            break
    
    # Same processing as part1
    activations = torch.cat(activations, dim=0)
    
    print("\n--- Pre-normalization activations ---")
    check_tensor(activations, "raw_activations", True)
    
    with torch.no_grad():
        norm_values = torch.norm(activations, dim=1).to(torch.float32)
        print("\n--- Activation norms percentiles ---")
        for p in [0, 0.1, 1, 5, 50, 95, 99, 99.9, 100]:
            percentile = torch.quantile(norm_values, torch.tensor(p/100, dtype=torch.float32)).item()
            print(f"Percentile {p}%: {percentile:.6f}")
    
    # Same clipping and normalization as part1
    with torch.no_grad():
        mean = activations.mean()
        std = activations.std()
        n_std = 6.0
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        below_count = (activations < lower_bound).sum().item()
        above_count = (activations > upper_bound).sum().item()
        total_elements = activations.numel()
        print(f"\nClipping bounds: {lower_bound.item()} to {upper_bound.item()}")
        print(f"Values below lower bound: {below_count} ({100.0 * below_count / total_elements:.6f}%)")
        print(f"Values above upper bound: {above_count} ({100.0 * above_count / total_elements:.6f}%)")
        
        activations = torch.clamp(activations, min=lower_bound, max=upper_bound)
        print("\n--- After clipping extreme values ---")
        check_tensor(activations, "clipped_activations")
    
    with torch.no_grad():
        mean_norm = torch.norm(activations, dim=1).mean()
        if mean_norm > 0:
            scale = np.sqrt(activations.shape[1]) / mean_norm
            activations = activations * scale
        else:
            print("WARNING: Mean norm is zero or negative, skipping normalization")
    
    print("\n--- Post-normalization activations ---")
    check_tensor(activations, "normalized_activations", True)
    
    return activations

def compute_residual_errors(original_activations, sae_model):
    """Compute residual errors using the trained SAE from part1."""
    residual_errors = []
    
    # Process in batches to avoid memory issues
    batch_size = BATCH_SIZE
    num_batches = (len(original_activations) + batch_size - 1) // batch_size
    
    sae_model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Computing residual errors"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(original_activations))
            batch = original_activations[start_idx:end_idx].to(DEVICE)
            
            # Get SAE reconstruction
            _, sae_reconstruction = sae_model(batch)
            
            # Compute residual error
            residual = batch - sae_reconstruction
            residual_errors.append(residual.cpu())
    
    return torch.cat(residual_errors, dim=0)

def train_residual_sae(residual_sae, train_loader, val_loader, num_steps, checkpoint_dir):
    """Train the residual SAE."""
    optimizer = optim.Adam(residual_sae.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0, eps=1e-5)
    
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1.0,
        end_factor=0.0,
        total_iters=int(num_steps * 0.2)
    )
    
    l1_scheduler = lambda step: min(L1_LAMBDA * (step / (num_steps * 0.05)), L1_LAMBDA)
    
    # Enhanced metrics tracking for residual SAE
    metrics_history = {
        'steps': [],
        'total_loss': [],
        'reconstruction_loss': [],
        'l1_loss': [],
        'best_loss': float('inf'),
        'l0_sparsity': [],
        'val_total_loss': [],
        'val_reconstruction_loss': [],
        'val_l1_loss': [],
        'val_best_loss': float('inf'),
        'val_l0_sparsity': [],
        # New metrics for tracking combined reconstruction quality
        'combined_reconstruction_error': [],  # Error with SAE1 + SAE2
        'sae1_only_error': [],  # Error with SAE1 only
        'val_combined_reconstruction_error': [],
        'val_sae1_only_error': []
    }
    
    prevent_sleep()
    
    try:
        train_iterator = iter(train_loader)
        
        for step in tqdm(range(num_steps), desc="Training Residual SAE"):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            # Batch contains both original activations and residual errors
            original_activations = batch['original'].to(DEVICE)
            residual_errors = batch['residual'].to(DEVICE)
            
            if check_tensor(residual_errors, f"residual_batch_step_{step}", print_stats=False):
                print(f"Problematic residual batch at step {step}")
                continue
            
            # Forward pass on residual errors
            residual_features, residual_reconstruction = residual_sae(residual_errors)
            
            # Check outputs
            features_issue = check_tensor(residual_features, f"residual_features_step_{step}", print_stats=False)
            recon_issue = check_tensor(residual_reconstruction, f"residual_reconstruction_step_{step}", print_stats=False)
            
            if features_issue or recon_issue:
                print(f"NaN or Inf detected in residual forward pass at step {step}")
                continue
            
            # Compute losses
            try:
                # Reconstruction loss for residual SAE
                reconstruction_diff = (residual_reconstruction - residual_errors)
                reconstruction_loss = torch.mean(reconstruction_diff ** 2)
                
                if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                    print(f"Reconstruction loss is {reconstruction_loss} at step {step}")
                    reconstruction_loss = torch.tensor(1.0, device=DEVICE)
                
                # L1 loss
                current_l1_lambda = l1_scheduler(step)
                decoder_norms = torch.norm(residual_sae.decoder.weight, p=2, dim=0)
                if check_tensor(decoder_norms, f"decoder_norms_step_{step}", print_stats=False):
                    decoder_norms = torch.ones_like(decoder_norms)
                
                abs_features = torch.abs(residual_features)
                if check_tensor(abs_features, f"abs_features_step_{step}", print_stats=False):
                    abs_features = torch.ones_like(abs_features)
                
                l1_loss = current_l1_lambda * torch.mean(abs_features * decoder_norms)
                
                if torch.isnan(l1_loss) or torch.isinf(l1_loss):
                    print(f"L1 loss is {l1_loss} at step {step}")
                    l1_loss = torch.tensor(0.0, device=DEVICE)
                
                # L0 sparsity
                l0_sparsity = compute_l0_sparsity(residual_features)
                
                # Total loss
                loss = reconstruction_loss + l1_loss
                loss = torch.clamp(loss, max=1e6)
                
                # Compute additional error metrics
                # SAE1 only error (this is just the residual error we're trying to model)
                sae1_only_error = torch.mean(residual_errors ** 2).item()
                
                # Combined error (original - (sae1_reconstruction + residual_sae_reconstruction))
                # Since residual_errors = original - sae1_reconstruction, 
                # the combined error is: original - (sae1_reconstruction + residual_reconstruction)
                # = original - sae1_reconstruction - residual_reconstruction
                # = residual_errors - residual_reconstruction
                combined_error = torch.mean((residual_errors - residual_reconstruction) ** 2).item()
                
            except Exception as e:
                print(f"Exception during loss computation at step {step}: {str(e)}")
                continue
            
            # Backward pass
            try:
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient checking and clipping
                for name, param in residual_sae.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"NaN/Inf detected in gradients for {name} at step {step}")
                            param.grad = torch.zeros_like(param.grad)
                
                optimizer.step()
                scheduler.step()
                
            except Exception as e:
                print(f"Exception during backward pass at step {step}: {str(e)}")
                continue
            
            # Track metrics
            metrics_history['steps'].append(step)
            metrics_history['total_loss'].append(loss.item())
            metrics_history['reconstruction_loss'].append(reconstruction_loss.item())
            metrics_history['l1_loss'].append(l1_loss.item())
            metrics_history['l0_sparsity'].append(l0_sparsity)
            metrics_history['combined_reconstruction_error'].append(combined_error)
            metrics_history['sae1_only_error'].append(sae1_only_error)
            
            if loss.item() < metrics_history['best_loss']:
                metrics_history['best_loss'] = loss.item()
            
            # Validation
            if step % VALIDATION_INTERVAL == 0:
                residual_sae.eval()
                val_metrics = validate_residual_sae(residual_sae, val_loader, current_l1_lambda)
                residual_sae.train()
                
                # Record validation metrics
                metrics_history['val_total_loss'].append(val_metrics['total_loss'])
                metrics_history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
                metrics_history['val_l1_loss'].append(val_metrics['l1_loss'])
                metrics_history['val_l0_sparsity'].append(val_metrics['l0_sparsity'])
                metrics_history['val_combined_reconstruction_error'].append(val_metrics['combined_reconstruction_error'])
                metrics_history['val_sae1_only_error'].append(val_metrics['sae1_only_error'])
                
                if val_metrics['total_loss'] < metrics_history['val_best_loss']:
                    metrics_history['val_best_loss'] = val_metrics['total_loss']
            
            # Print metrics
            if step % 1000 == 0:
                print(f"\nStep {step}")
                print(f"Train - Total Loss: {loss.item():.4f}")
                print(f"Train - Recon Loss: {reconstruction_loss.item():.4f}")
                print(f"Train - L1 Loss: {l1_loss.item():.4f}")
                print(f"Train - L0 Sparsity: {l0_sparsity:.4f}")
                print(f"Train - SAE1 Only Error: {sae1_only_error:.4f}")
                print(f"Train - Combined (SAE1+SAE2) Error: {combined_error:.4f}")
                print(f"Train - Error Reduction: {((sae1_only_error - combined_error) / sae1_only_error * 100):.2f}%")
                
                if metrics_history['val_total_loss']:
                    recent_val_idx = len(metrics_history['val_total_loss']) - 1
                    val_sae1_error = metrics_history['val_sae1_only_error'][recent_val_idx]
                    val_combined_error = metrics_history['val_combined_reconstruction_error'][recent_val_idx]
                    print(f"Val - Total Loss: {metrics_history['val_total_loss'][recent_val_idx]:.4f}")
                    print(f"Val - SAE1 Only Error: {val_sae1_error:.4f}")
                    print(f"Val - Combined (SAE1+SAE2) Error: {val_combined_error:.4f}")
                    print(f"Val - Error Reduction: {((val_sae1_error - val_combined_error) / val_sae1_error * 100):.2f}%")
                
                if step % 10000 == 0:
                    debug_model_parameters(residual_sae, f"residual_sae_at_step_{step}")
            
            # Save checkpoint
            if (step + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    residual_sae, optimizer, scheduler, step + 1,
                    metrics_history['val_best_loss'], metrics_history,
                    checkpoint_dir
                )
                print(f"\nCheckpoint saved at step {step + 1}")
    
    finally:
        allow_sleep()

def validate_residual_sae(residual_sae, val_loader, current_l1_lambda):
    """Compute validation metrics for the residual SAE."""
    val_metrics = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'l1_loss': 0.0,
        'l0_sparsity': 0.0,
        'combined_reconstruction_error': 0.0,
        'sae1_only_error': 0.0
    }
    
    max_val_batches = min(10, len(val_loader))
    val_iterator = iter(val_loader)
    num_valid_batches = 0
    
    with torch.no_grad():
        for _ in range(max_val_batches):
            try:
                batch = next(val_iterator)
                original_activations = batch['original'].to(DEVICE)
                residual_errors = batch['residual'].to(DEVICE)
                
                if check_tensor(residual_errors, "val_residual_batch", print_stats=False):
                    continue
                
                # Forward pass
                residual_features, residual_reconstruction = residual_sae(residual_errors)
                
                if (torch.isnan(residual_features).any() or torch.isinf(residual_features).any() or
                    torch.isnan(residual_reconstruction).any() or torch.isinf(residual_reconstruction).any()):
                    continue
                
                # Compute losses
                reconstruction_loss = torch.mean((residual_reconstruction - residual_errors) ** 2).item()
                
                decoder_norms = torch.norm(residual_sae.decoder.weight, p=2, dim=0)
                l1_loss = current_l1_lambda * torch.mean(torch.abs(residual_features) * decoder_norms).item()
                
                total_loss = reconstruction_loss + l1_loss
                l0_sparsity = compute_l0_sparsity(residual_features)
                
                # Error metrics
                sae1_only_error = torch.mean(residual_errors ** 2).item()
                combined_error = torch.mean((residual_errors - residual_reconstruction) ** 2).item()
                
                # Accumulate metrics
                val_metrics['total_loss'] += total_loss
                val_metrics['reconstruction_loss'] += reconstruction_loss
                val_metrics['l1_loss'] += l1_loss
                val_metrics['l0_sparsity'] += l0_sparsity
                val_metrics['combined_reconstruction_error'] += combined_error
                val_metrics['sae1_only_error'] += sae1_only_error
                
                num_valid_batches += 1
                
            except StopIteration:
                break
    
    # Average metrics
    if num_valid_batches > 0:
        for key in val_metrics:
            val_metrics[key] /= num_valid_batches
    else:
        print("Warning: No valid batches during validation!")
    
    return val_metrics

def main():
    parser = argparse.ArgumentParser(description='Train Residual SAE')
    parser.add_argument('--sae1_model_path', type=str, default='checkpoints/best_model.pt',
                        help='Path to the trained SAE from part1 (default: checkpoints/best_model.pt)')
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='Directory to save residual SAE checkpoints')
    
    args = parser.parse_args()
    
    # Set numerical stability
    torch.set_default_dtype(torch.float32)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=False, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype='auto', device_map='auto', local_files_only=True)
    
    # Collect activations (same as part1)
    print("Collecting activations...")
    activations = collect_activations(model, tokenizer, NUM_TOKENS)
    
    # Load the trained SAE from part1
    print(f"Loading trained SAE from {args.sae1_model_path}...")
    sae1 = SparseAutoencoder(
        input_dim=activations.shape[1],
        hidden_dim=NUM_FEATURES
    ).to(DEVICE)
    
    # Convert to same dtype as model
    sae1 = sae1.to(dtype=next(model.parameters()).dtype)
    
    # Load the trained weights
    if not os.path.exists(args.sae1_model_path):
        raise FileNotFoundError(f"SAE model not found at {args.sae1_model_path}. Please run part1 first.")
    
    sae1.load_state_dict(torch.load(args.sae1_model_path, map_location=DEVICE))
    print("SAE1 loaded successfully!")
    
    # Compute residual errors
    print("Computing residual errors...")
    residual_errors = compute_residual_errors(activations, sae1)
    
    print(f"Residual errors computed. Shape: {residual_errors.shape}")
    check_tensor(residual_errors, "residual_errors", True)
    
    # Create train/validation split
    print("Creating train/validation split...")
    dataset_size = len(activations)
    val_size = int(dataset_size * VALIDATION_SPLIT)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_original = activations[train_indices]
    train_residual = residual_errors[train_indices]
    val_original = activations[val_indices]
    val_residual = residual_errors[val_indices]
    
    print(f"Train set size: {train_size} samples")
    print(f"Validation set size: {val_size} samples")
    
    # Create datasets
    train_dataset = ResidualDataset(train_original, train_residual)
    val_dataset = ResidualDataset(val_original, val_residual)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize residual SAE (same architecture as part1)
    print("Initializing Residual SAE...")
    residual_sae = SparseAutoencoder(
        input_dim=activations.shape[1],  # Same input dimension
        hidden_dim=NUM_FEATURES  # Same number of features
    ).to(DEVICE)
    
    # Convert to same dtype as model
    residual_sae = residual_sae.to(dtype=next(model.parameters()).dtype)
    
    # Debug initial parameters
    debug_model_parameters(residual_sae, "initial_residual_sae")
    
    print("Training Residual SAE...")
    train_residual_sae(residual_sae, train_loader, val_loader, TRAIN_STEPS, args.checkpoint_dir)
    
    # Post-process: normalize decoder columns to unit norm
    with torch.no_grad():
        decoder_norms = torch.norm(residual_sae.decoder.weight, p=2, dim=0)
        decoder_norms = torch.clamp(decoder_norms, min=1e-8)
        residual_sae.decoder.weight.data = residual_sae.decoder.weight.data / decoder_norms
    
    # Save the final trained residual SAE
    final_model_path = "residual_sae_model.pt"
    torch.save(residual_sae.state_dict(), final_model_path)
    print(f"Training complete! Residual SAE saved as {final_model_path}")
    
    # Compute final error comparison on validation set
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    
    sae1.eval()
    residual_sae.eval()
    
    with torch.no_grad():
        # Take a sample of validation data for final comparison
        sample_size = min(1000, len(val_original))
        sample_indices = torch.randperm(len(val_original))[:sample_size]
        sample_original = val_original[sample_indices].to(DEVICE)
        sample_residual = val_residual[sample_indices].to(DEVICE)
        
        # SAE1 reconstruction
        _, sae1_reconstruction = sae1(sample_original)
        sae1_error = torch.mean((sample_original - sae1_reconstruction) ** 2).item()
        
        # Residual SAE reconstruction
        _, residual_reconstruction = residual_sae(sample_residual)
        
        # Combined reconstruction (SAE1 + Residual SAE)
        combined_reconstruction = sae1_reconstruction + residual_reconstruction
        combined_error = torch.mean((sample_original - combined_reconstruction) ** 2).item()
        
        # Calculate improvement
        error_reduction = ((sae1_error - combined_error) / sae1_error) * 100
        
        print(f"Original data MSE (baseline): {torch.mean(sample_original ** 2).item():.6f}")
        print(f"SAE1 only reconstruction error: {sae1_error:.6f}")
        print(f"SAE1 + Residual SAE reconstruction error: {combined_error:.6f}")
        print(f"Error reduction: {error_reduction:.2f}%")
        print(f"Relative error (SAE1 only): {(sae1_error / torch.mean(sample_original ** 2).item()) * 100:.2f}%")
        print(f"Relative error (Combined): {(combined_error / torch.mean(sample_original ** 2).item()) * 100:.2f}%")

if __name__ == "__main__":
    main()
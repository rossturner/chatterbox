"""
GRPO V3 Checkpoints - Model saving and loading utilities
"""

import copy
from pathlib import Path
from typing import Dict, Union
import torch

from .config import *
from .models import LoRALayer, merge_lora_weights


def save_checkpoint(
    model,
    lora_layers: Dict[str, LoRALayer],
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    val_reward: float,
    checkpoint_dir: Union[str, Path],
    is_best: bool = False
):
    """Enhanced checkpoint saving with comprehensive state preservation"""
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Collect LoRA state dicts
    lora_state_dict = {}
    for name, lora_layer in lora_layers.items():
        lora_state_dict[name] = lora_layer.state_dict()
    
    # Create comprehensive checkpoint
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'val_reward': val_reward,
        'lora_state_dict': lora_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'lora_config': {
            'rank': LORA_RANK,
            'alpha': LORA_ALPHA,
            'dropout': LORA_DROPOUT
        },
        'training_config': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'kl_coeff': KL_COEFF,
            'reward_normalization': REWARD_NORMALIZATION,
        }
    }
    
    if is_best:
        # Save best model checkpoint
        best_path = checkpoint_dir / BEST_MODEL_FILENAME
        torch.save(checkpoint, best_path)
        print(f"💾 Best checkpoint saved: {best_path}")
    else:
        # Save regular epoch checkpoint
        checkpoint_path = checkpoint_dir / f"grpo_v3_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")


def create_merged_model(model, lora_layers: Dict[str, LoRALayer]):
    """Create merged model with LoRA weights integrated"""
    
    # Create a deep copy of the model
    merged_model = copy.deepcopy(model)
    
    # Merge LoRA weights into the base model
    merge_lora_weights(merged_model, lora_layers)
    
    return merged_model


def save_merged_model(model, save_dir: Path):
    """Save complete merged model"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the complete model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': getattr(model, 'config', {}),
    }, save_dir / "pytorch_model.bin")
    
    print(f"Merged model saved to {save_dir}")


def load_checkpoint(model, lora_layers: Dict[str, LoRALayer], optimizer, checkpoint_path: Union[str, Path]):
    """Load checkpoint and restore training state"""
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load LoRA states
    if 'lora_state_dict' in checkpoint:
        for name, lora_layer in lora_layers.items():
            if name in checkpoint['lora_state_dict']:
                lora_layer.load_state_dict(checkpoint['lora_state_dict'][name])
                print(f"✅ Loaded LoRA state for: {name}")
    
    # Load optimizer state
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Loaded optimizer state")
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    val_reward = checkpoint.get('val_reward', 0.0)
    
    print(f"✅ Checkpoint loaded: epoch {epoch}, step {step}, val_reward {val_reward:.4f}")
    
    return epoch, step, val_reward
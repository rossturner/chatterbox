"""
GRPO V3 Models - LoRA implementation and model utilities
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List

from .config import *


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = x @ self.lora_A.T
        if self.dropout is not None:
            result = self.dropout(result)
        result = result @ self.lora_B.T
        return result * self.scaling


def inject_lora_layers(model: nn.Module, target_modules: List[str], rank: int, alpha: float, dropout: float):
    lora_layers = {}
    device = next(model.parameters()).device
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                lora = LoRALayer(module.in_features, module.out_features, rank, alpha, dropout)
                lora.to(device)
                lora_layers[name] = lora
                
                # Store original forward method
                original_forward = module.forward
                
                # Create new forward method that includes LoRA
                def make_lora_forward(orig_forward, lora_layer):
                    def lora_forward(x):
                        orig_out = orig_forward(x)
                        lora_out = lora_layer(x)
                        return orig_out + lora_out
                    return lora_forward
                
                # Replace forward method
                module.forward = make_lora_forward(original_forward, lora)
                
                print(f"✓ Injected LoRA layer: {name}")
    
    return lora_layers


def merge_lora_weights(model, lora_layers: Dict[str, LoRALayer]):
    """Merge LoRA weights into the base model"""
    
    for name, module in model.named_modules():
        if name in lora_layers and isinstance(module, nn.Linear):
            lora_layer = lora_layers[name]
            
            # Compute LoRA weight delta
            lora_weight = (lora_layer.lora_B @ lora_layer.lora_A) * lora_layer.scaling
            
            # Merge with base weights
            with torch.no_grad():
                module.weight.data += lora_weight
            
            print(f"✅ Merged LoRA weights for: {name}")


def save_lora_adapter(lora_layers: Dict[str, LoRALayer], filepath: str):
    """Save LoRA adapter with enhanced configuration"""
    
    lora_state_dict = {}
    for name, lora_layer in lora_layers.items():
        lora_state_dict[name] = lora_layer.state_dict()
    
    adapter_config = {
        'lora_state_dict': lora_state_dict,
        'config': {
            'rank': LORA_RANK,
            'alpha': LORA_ALPHA,
            'dropout': LORA_DROPOUT,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        },
        'training_args': {
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'kl_coeff': KL_COEFF,
            'reward_normalization': REWARD_NORMALIZATION
        }
    }
    
    torch.save(adapter_config, filepath)
    print(f"LoRA adapter saved: {filepath}")
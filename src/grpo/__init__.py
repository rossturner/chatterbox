"""
GRPO V3 Fast - Modular implementation for easy editing and maintenance
"""

from .config import *
from .models import LoRALayer, inject_lora_layers
from .dataset import PairedAudioDataset, AudioSample, load_audio_samples_from_pairs
from .rewards import compute_rewards
from .training import compute_grpo_loss
from .utils import (
    safe_tensor_index,
    safe_gather,
    validate_tensor_operation,
    normalize_audio_sample_rate,
    normalize_rewards,
    prepare_batch_conditionals,
    generate_samples,
    collate_fn
)
from .metrics import GRPOMetricsTracker
from .checkpoints import save_checkpoint, create_merged_model, save_merged_model

__all__ = [
    'LoRALayer', 'inject_lora_layers',
    'PairedAudioDataset', 'AudioSample', 'load_audio_samples_from_pairs',
    'compute_rewards', 'compute_grpo_loss',
    'GRPOMetricsTracker',
    'save_checkpoint', 'create_merged_model', 'save_merged_model',
]
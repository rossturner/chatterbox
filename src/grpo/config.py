"""
GRPO V3 Configuration - All training parameters in one place
"""

import torch

# ========== ENHANCED CONFIGURATION ==========
AUDIO_DATA_DIR = "./audio_data_v2"
TARGET_SAMPLE_RATE = 48000  # Normalize all audio to 48kHz
S3_SR = 16000  # Voice encoder sample rate

# GRPO Training Configuration with Professional Settings
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 4
EPOCHS = 6  # Extended GRPO training with per-epoch checkpointing
LEARNING_RATE = 5e-6  # More conservative than 1e-5
WARMUP_STEPS = 50  # Reduced from 500, ~2% of total updates
WEIGHT_DECAY = 0  # Explicit weight decay
OPTIMIZER_BETAS = (0.9, 0.95)  # Enhanced optimizer betas

# Early Stopping Configuration
EARLY_STOPPING_PATIENCE = 4  # Stop after 4 epochs without improvement
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement to count as progress

# Audio length constraints (keep natural speech boundaries)
MAX_AUDIO_LENGTH = 20.0  # Slightly increased to accommodate longest clips (17s)
MIN_AUDIO_LENGTH = 0.5  # Slightly reduced to include shortest clips (0.67s)

# LoRA Configuration
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Checkpointing - Epoch-based instead of step-based
SAVE_EVERY_EPOCH = True
EVAL_INTERVAL_UPDATES = 250  # Regular evaluation
CHECKPOINT_DIR = "checkpoints_grpo_v3"

# System Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TEXT_LENGTH = 1000
VALIDATION_SPLIT = 0.1

# Enhanced GRPO Configuration
NUM_SAMPLES_PER_INPUT = 1  # Reduced to 1 for faster training (still doubled internally)
KL_COEFF = 0.05  # Increased KL coefficient for stronger regularization (5x)
REWARD_NORMALIZATION = True  # Enable reward normalization for stability
REWARD_BASELINE_MOMENTUM = 0.99  # Momentum for baseline reward averaging
SPEAKER_SIM_WEIGHT = 1.0
LENGTH_PENALTY_WEIGHT = -0.5

# Memory Optimization
FP16 = True  # Enable mixed precision
GRADIENT_CHECKPOINTING = False  # Disabled - conflicts with model cache

# Early Stopping Tracking
BEST_MODEL_FILENAME = "best_grpo_v3_model.pt"  # Best model checkpoint name
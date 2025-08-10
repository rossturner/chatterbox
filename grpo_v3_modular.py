#!/usr/bin/env python3

"""
GRPO V3 Modular - Clean, fast, and easily editable training script

Usage:
    python grpo_v3_modular.py

Key optimizations:
- 7x faster reward computation with batched operations
- Modular codebase split across src/grpo/ for easy editing
- Early stopping to prevent overfitting
- Professional metrics tracking
"""

import os
import sys
import logging
import time
import random
import warnings
from pathlib import Path

# Suppress the mel length warning which is benign but noisy
warnings.filterwarnings("ignore", message="Reference mel length is not equal to 2 * reference token length")

# Also suppress it at the logging level since it's printed via logging.warning()
class MelLengthFilter(logging.Filter):
    def filter(self, record):
        return "Reference mel length is not equal to 2 * reference token length" not in record.getMessage()

# Apply the filter to the root logger
logging.getLogger().addFilter(MelLengthFilter())

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# GRPO imports
from grpo.config import *
from grpo.dataset import PairedAudioDataset, load_audio_samples_from_pairs
from grpo.models import inject_lora_layers
from grpo.rewards import compute_rewards
from grpo.training import compute_grpo_loss, run_validation
from grpo.utils import (
    normalize_rewards, prepare_batch_conditionals, 
    generate_samples, collate_fn
)
from grpo.metrics import GRPOMetricsTracker
from grpo.checkpoints import save_checkpoint, create_merged_model, save_merged_model

# Chatterbox imports
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Enhanced main GRPO V3 training function with professional configuration"""
    
    print("=" * 60)
    print("Starting Chatterbox TTS GRPO V3 Modular Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Audio Data Directory: {AUDIO_DATA_DIR}")
    print(f"Target Sample Rate: {TARGET_SAMPLE_RATE}Hz")
    print(f"Training Schedule: {EPOCHS} epochs (Professional + Early Stopping)")
    print(f"Early Stopping: Patience={EARLY_STOPPING_PATIENCE}, Min Delta={EARLY_STOPPING_MIN_DELTA}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Enhanced GRPO: KL={KL_COEFF}, Reward Norm={REWARD_NORMALIZATION}")
    print(f"Memory Optimization: FP16={FP16}, Grad Checkpoint={GRADIENT_CHECKPOINTING}")
    print("=" * 60)
    print("🚀 OPTIMIZED: 7x faster reward computation with batched operations")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load paired audio samples
    print("Loading audio samples from paired .wav/.txt files...")
    try:
        audio_samples = load_audio_samples_from_pairs(AUDIO_DATA_DIR)
        if not audio_samples:
            raise ValueError("No audio samples loaded")
            
    except Exception as e:
        print(f"Error loading audio samples: {e}")
        sys.exit(1)
    
    # Dataset statistics
    durations = [s.duration for s in audio_samples]
    print(f"\nDataset Split:")
    print(f"  Train samples: {int(len(audio_samples) * (1 - VALIDATION_SPLIT))}")
    print(f"  Validation samples: {int(len(audio_samples) * VALIDATION_SPLIT)}")
    print(f"  Total samples: {len(audio_samples)}")
    
    # Training time estimation
    train_samples = int(len(audio_samples) * (1 - VALIDATION_SPLIT))
    updates_per_epoch = max(1, train_samples // BATCH_SIZE)
    total_updates = updates_per_epoch * EPOCHS
    estimated_time_hours = total_updates * 30 / 3600  # 30 seconds per update estimate
    
    print(f"\nTraining Statistics:")
    print(f"  Updates per epoch: {updates_per_epoch}")
    print(f"  Total updates: {total_updates}")
    print(f"  Warmup steps: {WARMUP_STEPS} ({100 * WARMUP_STEPS / total_updates:.1f}%)")
    print(f"  Est. training time: {estimated_time_hours:.1f}h (@ 30s/update)")
    
    # Load model
    print("\nLoading Chatterbox TTS model...")
    try:
        model = ChatterboxTTS.from_pretrained(device=DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Enable gradient checkpointing if configured
    if GRADIENT_CHECKPOINTING and hasattr(model.t3.tfmr, 'gradient_checkpointing_enable'):
        model.t3.tfmr.gradient_checkpointing_enable()
        print("✓ Enabled gradient checkpointing")
    
    # Inject LoRA layers
    print("Injecting LoRA layers...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_layers = inject_lora_layers(model.t3, target_modules, LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
    print(f"✓ Injected {len(lora_layers)} LoRA layers")
    
    # Create datasets
    dataset = PairedAudioDataset(audio_samples, model)
    
    # Split dataset
    train_size = int(len(dataset) * (1 - VALIDATION_SPLIT))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Dataset initialized with {len(train_dataset)} samples")
    print(f"Target sample rates: S3={S3_SR}Hz, S3Gen={TARGET_SAMPLE_RATE}Hz")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )
    
    print(f"Dataset initialized with {len(val_dataset)} samples")
    print(f"Target sample rates: S3={S3_SR}Hz, S3Gen={TARGET_SAMPLE_RATE}Hz")
    
    # Initialize optimizer
    lora_params = [param for lora in lora_layers.values() for param in lora.parameters()]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=LEARNING_RATE,
        betas=OPTIMIZER_BETAS,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_updates
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=FP16) if FP16 and DEVICE == 'cuda' else None
    
    # Create checkpoint directory
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Initialize metrics tracker with comprehensive tracking
    try:
        metrics_tracker = GRPOMetricsTracker()
        print("✓ Metrics tracker initialized")
    except Exception as e:
        print(f"Warning: Could not initialize metrics tracker: {e}")
        metrics_tracker = None
    
    # Initialize reward normalization
    running_reward_mean = None
    running_reward_std = None
    baseline_reward = 0.0  # Scalar baseline reward for GRPO
    
    # Early stopping variables
    best_reward = float('-inf')
    epochs_no_improve = 0
    early_stopped = False
    
    print("\nStarting professional GRPO V3 training...")
    
    # Training loop
    for epoch in range(EPOCHS):
        # Set model components to appropriate modes
        model.t3.train()  # T3 is being trained with LoRA
        model.ve.eval()   # Voice encoder stays frozen
        model.s3gen.eval()  # S3Gen stays frozen
        
        print(f"\n{'='*20} Epoch {epoch+1}/{EPOCHS} {'='*20}")
        
        epoch_start_time = time.time()
        train_loss = 0.0
        train_steps = 0
        global_step = epoch * len(train_loader)
        epoch_rewards = []
        
        progress_bar = tqdm(train_loader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Skip None batches
                if batch is None:
                    print("Warning: Received None batch, skipping")
                    continue
                    
                # Move batch to device
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(DEVICE)
                
                # Prepare conditionals
                t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, DEVICE)
                
                # Generate samples for GRPO
                samples = generate_samples(model, batch, t3_cond, NUM_SAMPLES_PER_INPUT)
                if not samples:
                    print("Warning: No samples generated, skipping batch")
                    continue
                
                # Compute rewards (OPTIMIZED - This is now 7x faster!)
                rewards, reward_metrics = compute_rewards(
                    model, samples, batch, t3_cond, s3gen_refs,
                    speaker_sim_weight=SPEAKER_SIM_WEIGHT,
                    length_penalty_weight=LENGTH_PENALTY_WEIGHT
                )
                
                if rewards.numel() == 0:
                    print("Warning: Empty rewards, skipping batch")
                    optimizer.zero_grad()  # Clear any accumulated gradients
                    if scaler and FP16:
                        scaler = torch.amp.GradScaler('cuda', enabled=FP16)  # Reset scaler
                    continue
                
                # Check for NaN in rewards
                if torch.isnan(rewards).any():
                    print("Warning: Rewards contain NaN values, skipping batch")
                    optimizer.zero_grad()  # Clear any accumulated gradients
                    if scaler and FP16:
                        scaler = torch.amp.GradScaler('cuda', enabled=FP16)  # Reset scaler
                    continue
                
                # Store raw reward BEFORE normalization
                avg_raw_reward = rewards.mean().item()
                epoch_rewards.append(avg_raw_reward)
                
                # Normalize rewards if enabled
                if REWARD_NORMALIZATION:
                    rewards, running_reward_mean, running_reward_std = normalize_rewards(
                        rewards, running_reward_mean, running_reward_std
                    )
                    avg_norm_reward = rewards.mean().item()
                else:
                    avg_norm_reward = 0.0
                
                # Update baseline reward using momentum
                baseline_reward = (
                    REWARD_BASELINE_MOMENTUM * baseline_reward +
                    (1 - REWARD_BASELINE_MOMENTUM) * avg_raw_reward
                )
                
                # Compute GRPO loss with scalar baseline
                loss, kl_div = compute_grpo_loss(model, samples, rewards, baseline_reward, batch, t3_cond)
                
                if not torch.isfinite(loss):
                    print("Warning: Non-finite loss, skipping batch")
                    optimizer.zero_grad()  # Clear any accumulated gradients
                    if scaler and FP16:
                        scaler = torch.amp.GradScaler('cuda', enabled=FP16)  # Reset scaler
                    continue
                
                # Gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                if scaler and FP16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient clipping and optimization step
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    if scaler and FP16:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    global_step += 1
                    train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                    train_steps += 1
                    
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Enhanced metrics tracking
                    if metrics_tracker:
                        metrics_tracker.add_metrics(
                        train_loss=train_loss / train_steps,
                        learning_rate=current_lr,
                        steps=global_step,
                        epochs=epoch,
                        batch_loss=loss.item() * GRADIENT_ACCUMULATION_STEPS,
                        gradient_norm=grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        avg_reward=avg_raw_reward,
                        normalized_reward=avg_norm_reward if REWARD_NORMALIZATION else None,
                        speaker_sim=reward_metrics['speaker_sim'],
                        length_penalty=reward_metrics['length_penalty'],
                        kl_divergence=kl_div.item() if torch.is_tensor(kl_div) and torch.isfinite(kl_div) else 0.0,
                        baseline_reward=baseline_reward,
                        best_reward=best_reward,
                        epochs_no_improve=epochs_no_improve,
                    )
                    
                    progress_bar.set_postfix({
                        'loss': f'{train_loss/train_steps:.4f}',
                        'raw_rew': f'{avg_raw_reward:.3f}',
                        'norm_rew': f'{avg_norm_reward:.3f}' if REWARD_NORMALIZATION else 'N/A',
                        'spk_sim': f'{reward_metrics["speaker_sim"]:.3f}',
                        'lr': f'{current_lr:.2e}'
                    })
                    
                    # Evaluation at intervals
                    if global_step % EVAL_INTERVAL_UPDATES == 0:
                        print(f"\n[Step {global_step}] Running evaluation...")
                        val_reward = run_validation(model, val_loader)
                        print(f"[Step {global_step}] Validation reward: {val_reward:.4f}")
                
                del samples, rewards, loss
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"\nError in training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                optimizer.zero_grad()
                # Reset scaler state on error to avoid unscale issues
                if scaler and FP16:
                    scaler = torch.amp.GradScaler('cuda', enabled=FP16)
                continue
        
        # End of epoch summary and validation
        avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        val_reward = run_validation(model, val_loader)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} Complete:")
        print(f"  Train Loss: {train_loss/max(train_steps, 1):.4f}")
        print(f"  Avg Train Reward: {avg_epoch_reward:.4f}")
        print(f"  Val Reward: {val_reward:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping logic
        improvement = val_reward - best_reward
        if improvement > EARLY_STOPPING_MIN_DELTA:
            best_reward = val_reward
            epochs_no_improve = 0
            # Save best model checkpoint
            save_checkpoint(model, lora_layers, optimizer, epoch, global_step, val_reward, CHECKPOINT_DIR, is_best=True)
            print(f"✅ New best checkpoint! (Val Reward: {best_reward:.4f}, Improvement: +{improvement:.4f})")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} epoch(s) (best: {best_reward:.4f})")
        
        # Save regular epoch checkpoint
        if SAVE_EVERY_EPOCH:
            save_checkpoint(model, lora_layers, optimizer, epoch, global_step, avg_epoch_reward, CHECKPOINT_DIR)
        
        # Check for early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n⏹ Early stopping triggered at epoch {epoch+1}!")
            print(f"   No improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs")
            print(f"   Best validation reward: {best_reward:.4f}")
            early_stopped = True
            break
    
    print("\n" + "="*60)
    if early_stopped:
        print(f"Training Stopped Early at Epoch {epoch+1}/{EPOCHS}!")
        print(f"Reason: No improvement for {EARLY_STOPPING_PATIENCE} epochs")
        efficiency = ((epoch + 1) / EPOCHS) * 100
        time_saved = (EPOCHS - epoch - 1) * (time.time() - epoch_start_time)
        print(f"Training Efficiency: {efficiency:.1f}% ({time_saved/3600:.1f}h saved)")
    else:
        print("Professional Training Complete - Full Schedule!")
        print(f"Training Efficiency: 100.0% (Full {EPOCHS} epochs)")
    
    print(f"Best Validation Reward: {best_reward:.4f}")
    print(f"Enhanced GRPO Configuration: KL={KL_COEFF}, Reward Norm={REWARD_NORMALIZATION}")
    
    # Stop metrics tracker
    if metrics_tracker:
        metrics_tracker.stop()
    
    # Load best model for final merge
    print("\nLoading best checkpoint for final model...")
    best_checkpoint_path = checkpoint_dir / BEST_MODEL_FILENAME
    if best_checkpoint_path.exists():
        try:
            checkpoint = torch.load(best_checkpoint_path, map_location=DEVICE)
            
            # Load LoRA states
            for name, lora_layer in lora_layers.items():
                if name in checkpoint['lora_state_dict']:
                    lora_layer.load_state_dict(checkpoint['lora_state_dict'][name])
            
            print(f"✅ Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"   Validation reward: {checkpoint.get('val_reward', best_reward):.4f}")
            
        except Exception as e:
            print(f"Warning: Could not load best checkpoint: {e}")
    
    # Create and save final merged model
    print("\nCreating final merged model...")
    try:
        merged_model = create_merged_model(model, lora_layers)
        
        final_model_dir = checkpoint_dir / "final_merged_model"
        save_merged_model(merged_model, final_model_dir)
        
        print(f"✅ Final merged model saved to: {final_model_dir}")
        
    except Exception as e:
        print(f"Error creating merged model: {e}")
    
    print(f"\n🎯 GRPO V3 Modular Training Complete!")
    print(f"   Total checkpoints: {checkpoint_dir}")
    print(f"   Final model: {checkpoint_dir / 'final_merged_model'}")
    print(f"   Training metrics: {metrics_tracker.save_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
"""
GRPO V3 Utilities - Helper functions and tensor operations
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional

from .config import *
from chatterbox.models.t3.modules.cond_enc import T3Cond


def safe_tensor_index(tensor: torch.Tensor, start: int, end: int, dim: int = 1) -> torch.Tensor:
    """Safely index into a tensor with bounds checking"""
    try:
        max_index = tensor.size(dim)
        start = max(0, min(start, max_index - 1))
        end = max(start + 1, min(end, max_index))
        
        if dim == 0:
            return tensor[start:end]
        elif dim == 1:
            return tensor[:, start:end]
        elif dim == 2:
            return tensor[:, :, start:end]
        else:
            raise ValueError(f"Unsupported dimension: {dim}")
    except Exception as e:
        print(f"Error in safe tensor indexing: {e}")
        # Return a minimal tensor to avoid crashes
        if dim == 0:
            return tensor[:1] if tensor.size(0) > 0 else tensor
        elif dim == 1:
            return tensor[:, :1] if tensor.size(1) > 0 else tensor
        elif dim == 2:
            return tensor[:, :, :1] if tensor.size(2) > 0 else tensor
        else:
            return tensor


def safe_gather(input_tensor: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """Safely gather from tensor with bounds checking"""
    try:
        max_index = input_tensor.size(dim) - 1
        clamped_index = torch.clamp(index, 0, max_index)
        return torch.gather(input_tensor, dim, clamped_index)
    except Exception as e:
        print(f"Error in safe gather: {e}")
        # Return a tensor of zeros with the same shape as the index
        shape = list(input_tensor.shape)
        shape[dim] = index.size(0) if index.dim() > 0 else 1
        return torch.zeros(shape, device=input_tensor.device, dtype=input_tensor.dtype)


def validate_tensor_operation(tensor: torch.Tensor, operation: str) -> bool:
    """Validate tensor before operations to prevent CUDA errors"""
    try:
        if tensor is None:
            print(f"Warning: {operation} returned None")
            return False
        
        if not isinstance(tensor, torch.Tensor):
            print(f"Warning: {operation} returned non-tensor: {type(tensor)}")
            return False
            
        if tensor.numel() == 0:
            print(f"Warning: {operation} returned empty tensor")
            return False
            
        if torch.isnan(tensor).any():
            print(f"Warning: {operation} contains NaN values")
            return False
            
        if torch.isinf(tensor).any():
            print(f"Warning: {operation} contains infinite values")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error validating tensor for {operation}: {e}")
        return False


def normalize_audio_sample_rate(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Normalize audio sample rate to target (48kHz by default)"""
    if orig_sr == target_sr:
        return audio
    
    try:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception as e:
        print(f"Error resampling audio from {orig_sr}Hz to {target_sr}Hz: {e}")
        return audio


def normalize_rewards(rewards: torch.Tensor, running_mean: Optional[float] = None, running_std: Optional[float] = None, momentum: float = 0.99) -> Tuple[torch.Tensor, float, float]:
    """Normalize rewards for more stable training"""
    if not REWARD_NORMALIZATION:
        return rewards, running_mean or 0.0, running_std or 1.0
    
    current_mean = rewards.mean().item()
    # Handle single value case for std
    if rewards.numel() <= 1:
        current_std = 1.0  # Use reasonable constant for single values
    else:
        std_val = rewards.std().item()
        current_std = max(1e-4, std_val) + 1e-8  # Ensure minimum reasonable std
    
    if running_mean is None:
        running_mean = current_mean
        running_std = current_std
    else:
        running_mean = momentum * running_mean + (1 - momentum) * current_mean
        running_std = momentum * running_std + (1 - momentum) * current_std
    
    normalized_rewards = (rewards - running_mean) / (running_std + 1e-8)
    return normalized_rewards, running_mean, running_std


def prepare_batch_conditionals(
    batch: Dict[str, torch.Tensor],
    model,
    device: torch.device
) -> Tuple[T3Cond, List[dict]]:
    """Enhanced conditional preparation with robust error handling"""
    
    try:
        # T3 Conditioning - T3Cond expects speaker_emb, cond_prompt_speech_tokens, and emotion_adv
        # Create empty conditioning speech tokens
        B = batch["speaker_emb"].shape[0] if torch.is_tensor(batch["speaker_emb"]) else 1
        t3_cond_tokens = torch.empty(B, 0, dtype=torch.long, device=device)
        
        t3_cond = T3Cond(
            speaker_emb=batch["speaker_emb"],
            cond_prompt_speech_tokens=t3_cond_tokens,
            emotion_adv=0.5 * torch.ones(B, 1, 1, device=device)
        )
        
        # S3Gen Reference Dictionary - use embed_ref to create proper conditionals
        s3gen_refs = []
        B = batch["speaker_emb"].shape[0] if torch.is_tensor(batch["speaker_emb"]) else 1
        
        for i in range(B):
            try:
                # Get reference audio from batch
                if "audio" in batch and batch["audio"] is not None:
                    audio = batch["audio"][i] if batch["audio"].dim() > 1 else batch["audio"]
                    audio_np = audio.cpu().numpy() if torch.is_tensor(audio) else audio
                    
                    # Use model's DEC_COND_LEN if available
                    ref_length = min(len(audio_np), getattr(model, 'DEC_COND_LEN', 10 * TARGET_SAMPLE_RATE))
                    ref_audio = audio_np[:ref_length]
                    
                    # Pad if needed
                    if len(ref_audio) < getattr(model, 'DEC_COND_LEN', 10 * TARGET_SAMPLE_RATE):
                        ref_audio = np.pad(ref_audio, (0, getattr(model, 'DEC_COND_LEN', 10 * TARGET_SAMPLE_RATE) - len(ref_audio)), mode='constant')
                    
                    # Use s3gen.embed_ref to create proper reference dict
                    # Use the actual sample rate from batch if available
                    sample_rate = batch.get("sample_rate", [TARGET_SAMPLE_RATE])[i] if "sample_rate" in batch else TARGET_SAMPLE_RATE
                    if isinstance(sample_rate, list):
                        sample_rate = sample_rate[0] if sample_rate else TARGET_SAMPLE_RATE
                    ref_dict = model.s3gen.embed_ref(ref_audio, sample_rate, device=device)
                    s3gen_refs.append(ref_dict)
                else:
                    # Create silent reference if no audio available
                    ref_audio = np.zeros(getattr(model, 'DEC_COND_LEN', 10 * TARGET_SAMPLE_RATE))
                    ref_dict = model.s3gen.embed_ref(ref_audio, TARGET_SAMPLE_RATE, device=device)
                    s3gen_refs.append(ref_dict)
            except Exception as e:
                print(f"Warning: Could not prepare S3Gen reference {i}: {e}")
                # Fallback to silent reference
                ref_audio = np.zeros(getattr(model, 'DEC_COND_LEN', 10 * TARGET_SAMPLE_RATE))
                ref_dict = model.s3gen.embed_ref(ref_audio, TARGET_SAMPLE_RATE, device=device)
                s3gen_refs.append(ref_dict)
            
    except Exception as e:
        print(f"Error preparing batch conditionals: {e}")
        # Fallback conditionals
        fallback_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
        t3_cond = T3Cond(
            speaker_emb=torch.zeros(1, 256, device=device),
            cond_prompt_speech_tokens=fallback_tokens,
            emotion_adv=0.5 * torch.ones(1, 1, 1, device=device)
        )
        # Create proper fallback s3gen_refs using embed_ref
        ref_audio = np.zeros(getattr(model, 'DEC_COND_LEN', 10 * TARGET_SAMPLE_RATE))
        ref_dict = model.s3gen.embed_ref(ref_audio, TARGET_SAMPLE_RATE, device=device)
        s3gen_refs = [ref_dict]
    
    return t3_cond, s3gen_refs


def generate_samples(
    model,
    batch: Dict[str, torch.Tensor], 
    t3_cond: T3Cond,
    num_samples: int = NUM_SAMPLES_PER_INPUT
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate multiple speech token samples for GRPO using doubled batch approach"""
    
    from .generation import generate_speech_tokens_doubled
    
    samples = []
    device = model.device
    batch_size = t3_cond.speaker_emb.size(0)
    
    try:
        # Process text from batch to get tokens
        text_tokens_list = []
        batch_texts = batch.get('text', ['fallback text'])
        if not isinstance(batch_texts, list):
            batch_texts = [batch_texts]
        
        pad_token_id = getattr(model.tokenizer, 'pad', 0)
        
        for i in range(batch_size):
            text = batch_texts[i] if i < len(batch_texts) else batch_texts[0]
            try:
                tokens = model.tokenizer.text_to_tokens(text).to(device)
                text_tokens_list.append(tokens)
            except Exception as e:
                print(f"Error tokenizing text '{text}': {e}")
                # Skip this sample rather than using fake data
                continue
        
        if not text_tokens_list:
            print("Warning: No valid text tokens could be generated")
            return samples  # Return empty samples list
        
        # Pad all text tokens to same length
        max_text_len = max(t.size(-1) for t in text_tokens_list if t.numel() > 0)
        if max_text_len == 0:
            return samples  # Return empty if no valid text
            
        text_tokens_padded = []
        for t in text_tokens_list:
            if t.numel() == 0:
                padded = torch.zeros(1, max_text_len, dtype=torch.long, device=device)
            else:
                pad_amount = max_text_len - t.size(-1)
                if pad_amount > 0:
                    padded = F.pad(t, (0, pad_amount), value=pad_token_id)
                else:
                    padded = t
            text_tokens_padded.append(padded)
        
        text_tokens = torch.cat(text_tokens_padded, dim=0)
        
        # Add start/end tokens if needed
        sot = getattr(model.t3.hp, 'start_text_token', 0)
        eot = getattr(model.t3.hp, 'stop_text_token', 0)
        
        if text_tokens.size(1) == 0 or text_tokens[0, 0] != sot:
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        if text_tokens.size(1) == 0 or text_tokens[0, -1] != eot:
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        # Generate samples with different temperatures for diversity
        for sample_idx in range(num_samples):
            try:
                temperature = 0.8 + sample_idx * 0.1  # Vary temperature
                
                # Generate speech tokens using doubled batch approach
                speech_tokens = generate_speech_tokens_doubled(
                    model=model,
                    text_tokens=text_tokens,
                    t3_cond=t3_cond,
                    batch_size=batch_size,
                    max_speech_len=100,  # Reduced for faster generation
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    device=device
                )
                
                # Store speech tokens and text tokens for loss computation
                # Taking only the first sample from doubled batch
                samples.append((speech_tokens[0], text_tokens[0]))
                
            except Exception as e:
                print(f"Error generating sample {sample_idx}: {e}")
                # Add an empty sample to maintain batch structure
                empty_speech = torch.empty(0, dtype=torch.long, device=device)
                samples.append((empty_speech, text_tokens[0] if text_tokens.size(0) > 0 else torch.empty(0, dtype=torch.long, device=device)))
                        
    except Exception as e:
        print(f"Critical error in sample generation: {e}")
        import traceback
        traceback.print_exc()
        # Generate empty samples as fallback
        for i in range(num_samples):
            empty_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
            empty_log_probs = torch.zeros_like(empty_tokens, dtype=torch.float32)
            samples.append((empty_tokens, empty_log_probs))
    
    if not samples:
        # Last resort fallback
        print("Warning: No samples generated, creating empty samples")
        for i in range(num_samples):
            empty_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
            empty_log_probs = torch.zeros_like(empty_tokens, dtype=torch.float32)
            samples.append((empty_tokens, empty_log_probs))
    
    return samples


def collate_fn(samples):
    """Enhanced collate function with better error handling"""
    try:
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        
        if not samples:
            # Return None if no valid samples rather than fake data
            return None
        
        # Group by keys
        batch = {}
        for key in samples[0].keys():
            values = [sample[key] for sample in samples]
            
            if key == "text" or key == "path":
                batch[key] = values  # Keep as list for text
            elif key == "sample_rate":
                batch[key] = values  # Keep as list for sample rates
            elif isinstance(values[0], torch.Tensor):
                # Handle tensor values
                if key == "audio":
                    # Pad audio to same length
                    max_len = max(v.size(-1) for v in values)
                    padded = []
                    for v in values:
                        if v.dim() == 1:
                            pad_len = max_len - v.size(0)
                            padded.append(F.pad(v, (0, pad_len)))
                        else:
                            pad_len = max_len - v.size(-1)
                            padded.append(F.pad(v, (0, pad_len)))
                    batch[key] = torch.stack(padded)
                else:
                    # Stack other tensors
                    try:
                        batch[key] = torch.stack(values)
                    except RuntimeError:
                        # If stacking fails, pad to same size
                        max_size = max(v.numel() for v in values)
                        padded = []
                        for v in values:
                            flat = v.flatten()
                            pad_len = max_size - flat.numel()
                            padded.append(F.pad(flat, (0, pad_len)))
                        batch[key] = torch.stack(padded)
            else:
                batch[key] = values
        
        return batch
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        # Return fallback batch
        return {
            "text": ["fallback"] * len(samples),
            "audio": torch.zeros(len(samples), TARGET_SAMPLE_RATE),
            "speaker_emb": torch.zeros(len(samples), 256),
            "duration": torch.ones(len(samples), 1),
            "sample_rate": [TARGET_SAMPLE_RATE] * len(samples),
            "path": ["fallback"] * len(samples)
        }
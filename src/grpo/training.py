"""
GRPO V3 Training - GRPO loss computation and training utilities
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .config import *
from .utils import validate_tensor_operation
from chatterbox.models.t3.modules.cond_enc import T3Cond


def compute_grpo_loss(
    model,
    samples: List[Tuple[torch.Tensor, torch.Tensor]], 
    rewards: torch.Tensor,
    baseline_reward: float,
    batch: Dict[str, torch.Tensor],
    t3_cond: T3Cond,
    kl_coeff: float = KL_COEFF
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRPO loss computation matching the original grpo.py implementation
    """
    
    device = model.device
    batch_size = t3_cond.speaker_emb.size(0)  # Get actual batch size from t3_cond
    
    if not validate_tensor_operation(rewards, "rewards"):
        return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
    
    # Compute advantages using scalar baseline
    advantages = rewards - baseline_reward
    ranked_indices = torch.argsort(advantages, descending=True)
    
    total_loss = 0.0
    total_kl = 0.0
    valid_samples = 0
    
    with torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu', enabled=(device == 'cuda' and FP16), dtype=torch.float16):
        for rank, idx in enumerate(ranked_indices):
            try:
                speech_tokens, text_tokens = samples[idx]
                
                if not validate_tensor_operation(speech_tokens, f"speech tokens {idx}"):
                    continue
                    
                if not validate_tensor_operation(text_tokens, f"text tokens {idx}"):
                    continue
                
                if speech_tokens.numel() == 0:
                    continue
                
                text_tokens = text_tokens.unsqueeze(0).to(device) if text_tokens.dim() == 1 else text_tokens.to(device)
                speech_tokens = speech_tokens.to(device)
                
                # Validate token values are within bounds
                vocab_size = getattr(model.t3, 'speech_vocab_size', 1024)
                speech_tokens = torch.clamp(speech_tokens, 0, vocab_size - 1)
                
                # Ensure text_tokens is repeated for the batch size
                if text_tokens.size(0) == 1 and batch_size > 1:
                    text_tokens = text_tokens.repeat(batch_size, 1)
                
                # Double for the two samples per input
                text_tokens_doubled = torch.cat([text_tokens, text_tokens], dim=0)
                speech_tokens_doubled = torch.cat([speech_tokens, speech_tokens], dim=0) if speech_tokens.dim() == 2 else torch.cat([speech_tokens.unsqueeze(0), speech_tokens.unsqueeze(0)], dim=0)
                
                # Create doubled conditionals
                t3_cond_doubled = T3Cond(
                    speaker_emb=torch.cat([t3_cond.speaker_emb, t3_cond.speaker_emb], dim=0),
                    cond_prompt_speech_tokens=torch.cat(
                        [t3_cond.cond_prompt_speech_tokens, t3_cond.cond_prompt_speech_tokens], dim=0
                    ) if t3_cond.cond_prompt_speech_tokens.numel() > 0 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
                    emotion_adv=torch.cat(
                        [t3_cond.emotion_adv, t3_cond.emotion_adv], dim=0
                    ) if t3_cond.emotion_adv is not None else None
                )
                
                # Prepare input tokens (all but last for input)
                input_speech_tokens = speech_tokens_doubled[:, :-1] if speech_tokens_doubled.size(1) > 1 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device)
                
                embeds, len_cond = model.t3.prepare_input_embeds(
                    t3_cond=t3_cond_doubled,
                    text_tokens=text_tokens_doubled,
                    speech_tokens=input_speech_tokens,
                )
                
                if not validate_tensor_operation(embeds, f"embeds {idx}"):
                    continue
                
                # Only enable gradient checkpointing if configured
                if GRADIENT_CHECKPOINTING and hasattr(model.t3.tfmr, 'gradient_checkpointing_enable'):
                    model.t3.tfmr.gradient_checkpointing_enable()
                
                hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]
                
                if not validate_tensor_operation(hidden_states, f"hidden states {idx}"):
                    continue
                
                # Calculate speech portion bounds
                speech_start = len_cond + text_tokens_doubled.size(1)
                speech_end = min(speech_start + speech_tokens_doubled.size(1) - 1, hidden_states.size(1))
                
                if speech_start < speech_end and speech_start >= 0 and speech_end <= hidden_states.size(1):
                    from .utils import safe_tensor_index, safe_gather
                    speech_hidden = safe_tensor_index(hidden_states, speech_start, speech_end, dim=1)
                    
                    if speech_hidden.numel() > 0 and validate_tensor_operation(speech_hidden, f"speech hidden {idx}"):
                        speech_logits = model.t3.speech_head(speech_hidden)
                        
                        if not validate_tensor_operation(speech_logits, f"speech logits {idx}"):
                            continue
                        
                        # Prepare target tokens (shifted by 1 for prediction)
                        target_end = min(speech_end - speech_start + 1, speech_tokens_doubled.size(1) - 1)
                        target_tokens = safe_tensor_index(speech_tokens_doubled, 1, 1 + target_end, dim=1)
                        
                        if target_tokens.numel() > 0 and speech_logits.size(1) >= target_tokens.size(1):
                            speech_logits = speech_logits[:, :target_tokens.size(1)]
                            
                            # Clamp logits to prevent overflow
                            speech_logits = torch.clamp(speech_logits, -10.0, 10.0)
                            
                            log_probs = F.log_softmax(speech_logits, dim=-1)
                            
                            if not validate_tensor_operation(log_probs, f"log probs {idx}"):
                                continue
                            
                            # Take only the first sample from the doubled batch
                            gathered_log_probs = safe_gather(
                                log_probs[0],
                                -1,
                                target_tokens[0].unsqueeze(-1)
                            ).squeeze(-1)
                            
                            if gathered_log_probs.numel() > 0 and validate_tensor_operation(gathered_log_probs, f"gathered log probs {idx}"):
                                rank_weight = 1.0 / (rank + 1)
                                sample_loss = -gathered_log_probs.mean() * rank_weight * advantages[idx]
                                
                                if validate_tensor_operation(sample_loss, f"sample loss {idx}"):
                                    total_loss += sample_loss
                                    valid_samples += 1
                                
                                # Compute KL divergence safely
                                probs = log_probs[0].exp()
                                if validate_tensor_operation(probs, f"probs for KL {idx}"):
                                    kl_div = (probs * log_probs[0]).sum(-1).mean()
                                    if validate_tensor_operation(kl_div, f"kl div {idx}"):
                                        total_kl += kl_div
                
                del embeds, hidden_states
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error in GRPO loss computation for sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if valid_samples > 0:
        total_loss = total_loss / valid_samples + kl_coeff * total_kl / valid_samples
        total_kl = total_kl / valid_samples
    else:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl = torch.tensor(0.0, device=device)
    
    return total_loss, total_kl


def run_validation(model, val_loader) -> float:
    """Run validation and return average reward"""
    import numpy as np
    from .rewards import compute_rewards
    from .utils import prepare_batch_conditionals, generate_samples
    
    model.t3.eval()
    model.ve.eval() 
    model.s3gen.eval()
    val_rewards = []
    
    with torch.no_grad():
        for batch in val_loader:
            try:
                # Move batch to device
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch[key] = value.to(DEVICE)
                
                # Prepare conditionals
                t3_cond, s3gen_refs = prepare_batch_conditionals(batch, model, DEVICE)
                
                # Generate samples
                samples = generate_samples(model, batch, t3_cond, 1)  # Single sample for validation
                
                if samples:
                    # Compute rewards
                    rewards, _ = compute_rewards(
                        model, samples, batch, t3_cond, s3gen_refs,
                        speaker_sim_weight=SPEAKER_SIM_WEIGHT,
                        length_penalty_weight=LENGTH_PENALTY_WEIGHT
                    )
                    
                    if rewards.numel() > 0:
                        val_rewards.extend(rewards.cpu().numpy())
                        
            except Exception as e:
                print(f"Validation error: {e}")
                continue
    
    model.t3.train()
    model.ve.eval()
    model.s3gen.eval()
    return float(np.mean(val_rewards)) if val_rewards else 0.0
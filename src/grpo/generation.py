"""
GRPO V3 Generation - Proper speech token generation using T3 model
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from .config import *
from .utils import validate_tensor_operation
from chatterbox.models.t3.modules.cond_enc import T3Cond


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
    if logits.numel() == 0:
        return logits
    
    try:
        assert logits.dim() == 1  # batch size 1 for now
        top_k = min(top_k, logits.size(-1))  # Safety check
        
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
    except Exception as e:
        print(f"Error in top_k_top_p_filtering: {e}")
        
    return logits


def generate_speech_tokens_doubled(
    model,
    text_tokens: torch.Tensor,
    t3_cond: T3Cond,
    batch_size: int,
    max_speech_len: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = None
) -> torch.Tensor:
    """Generate speech tokens using doubled batch approach from grpo.py"""
    
    if device is None:
        device = model.device
    
    try:
        # Double everything for generating two samples
        text_tokens_doubled = torch.cat([text_tokens, text_tokens], dim=0)
        t3_cond_doubled = T3Cond(
            speaker_emb=torch.cat([t3_cond.speaker_emb, t3_cond.speaker_emb], dim=0),
            cond_prompt_speech_tokens=torch.cat(
                [t3_cond.cond_prompt_speech_tokens, t3_cond.cond_prompt_speech_tokens], dim=0
            ) if t3_cond.cond_prompt_speech_tokens.numel() > 0 else torch.empty(batch_size * 2, 0, dtype=torch.long, device=device),
            emotion_adv=torch.cat(
                [t3_cond.emotion_adv, t3_cond.emotion_adv], dim=0
            ) if t3_cond.emotion_adv is not None else None
        )
        
        # Initialize with empty speech tokens
        empty_speech = torch.empty(batch_size * 2, 0, dtype=torch.long, device=device)
        
        # Prepare initial embeddings
        embeds, len_cond = model.t3.prepare_input_embeds(
            t3_cond=t3_cond_doubled,
            text_tokens=text_tokens_doubled,
            speech_tokens=empty_speech,
        )
        
        if not validate_tensor_operation(embeds, "initial embeds"):
            return torch.empty(batch_size * 2, 0, dtype=torch.long, device=device)
        
        generated_tokens = []
        max_context_len = 512
        vocab_size = getattr(model.t3, 'speech_vocab_size', 1024)
        
        for step in range(max_speech_len):
            # Truncate if too long
            if embeds.size(1) > max_context_len:
                embeds = embeds[:, -max_context_len:]
            
            if not validate_tensor_operation(embeds, f"embeds step {step}"):
                break
            
            # Forward through transformer
            hidden_states = model.t3.tfmr(inputs_embeds=embeds)[0]
            
            if not validate_tensor_operation(hidden_states, f"hidden states step {step}"):
                break
            
            if hidden_states.size(1) == 0:
                break
            
            # Get speech logits
            speech_logits = model.t3.speech_head(hidden_states[:, -1:])
            
            if not validate_tensor_operation(speech_logits, f"speech logits step {step}"):
                break
            
            # Apply temperature
            speech_logits = speech_logits / temperature
            
            # Apply top-k top-p filtering
            filtered_logits = top_k_top_p_filtering(
                speech_logits[0, 0], 
                top_k=top_k, 
                top_p=top_p
            )
            
            # Clamp logits to prevent overflow
            filtered_logits = torch.clamp(filtered_logits, -10.0, 10.0)
            
            # Sample next token
            probs = F.softmax(filtered_logits, dim=-1)
            
            if not validate_tensor_operation(probs, f"probs step {step}"):
                break
            
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Validate token is within vocab bounds
            if next_token.item() >= vocab_size:
                next_token = torch.tensor([vocab_size - 1], device=device, dtype=torch.long)
            
            # Check for stop token
            if hasattr(model.t3, 'hp') and hasattr(model.t3.hp, 'stop_speech_token'):
                if next_token.item() == model.t3.hp.stop_speech_token:
                    break
            
            generated_tokens.append(next_token)
            
            # Get speech embedding for next step
            if hasattr(model.t3, 'speech_embed'):
                next_embed = model.t3.speech_embed(next_token.unsqueeze(0).expand(batch_size * 2, -1))
            elif hasattr(model.t3, 'speech_emb'):
                next_embed = model.t3.speech_emb(next_token.unsqueeze(0).expand(batch_size * 2, -1))
            else:
                print("Warning: Could not find speech embedding layer")
                break
            
            if not validate_tensor_operation(next_embed, f"next embed step {step}"):
                break
            
            # Append to embeddings
            embeds = torch.cat([embeds, next_embed], dim=1)
        
        # Combine generated tokens
        if generated_tokens:
            speech_tokens = torch.cat(generated_tokens, dim=0).unsqueeze(0)
        else:
            speech_tokens = torch.empty(1, 0, dtype=torch.long, device=device)
        
        return speech_tokens
        
    except Exception as e:
        print(f"Error in generate_speech_tokens_doubled: {e}")
        import traceback
        traceback.print_exc()
        return torch.empty(batch_size * 2, 0, dtype=torch.long, device=device)


def generate_speech_tokens(
    model,
    text_tokens: torch.Tensor,
    t3_cond: T3Cond,
    max_speech_len: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = None
) -> torch.Tensor:
    """Single sample generation for validation"""
    
    if device is None:
        device = model.device
    
    # Ensure text_tokens has batch dimension
    if text_tokens.dim() == 1:
        text_tokens = text_tokens.unsqueeze(0)
    
    batch_size = text_tokens.size(0)
    
    # Use doubled generation and return first sample
    doubled_result = generate_speech_tokens_doubled(
        model=model,
        text_tokens=text_tokens,
        t3_cond=t3_cond,
        batch_size=batch_size,
        max_speech_len=max_speech_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device
    )
    
    # Return only the first sample
    return doubled_result[0:1] if doubled_result.size(0) > 0 else torch.empty(1, 0, dtype=torch.long, device=device)
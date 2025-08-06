# Better Performance Optimizations for Chatterbox Streaming

## Context Window Considerations

**Don't reduce context_window below 50 tokens** - this maintains audio quality by:
- Preserving prosody and intonation across chunks
- Preventing audio artifacts at chunk boundaries
- Ensuring consistent voice characteristics

## Recommended Optimization Priorities

### 1. **Model Compilation (Highest Impact)**
```python
# Add to ChatterboxTTS.__init__ or from_pretrained
if hasattr(torch, 'compile'):
    self.t3.tfmr = torch.compile(self.t3.tfmr, mode='reduce-overhead')
    self.s3gen = torch.compile(self.s3gen, mode='reduce-overhead')
```

### 2. **Optimal Chunk Configuration**
```python
# Balanced performance vs quality - IMPORTANT: context_window >= 2x chunk_size
model.generate_stream(
    text=text,
    chunk_size=75,           # Sweet spot for efficiency
    context_window=150,      # 2x chunk_size for proper continuity
    temperature=0.8,
    cfg_weight=0.5
)
```

### 3. **KV Cache Optimization**
The deprecation warnings about tuple format suggest updating to modern cache:
```python
# In transformer calls, use modern cache format
from transformers import DynamicCache
past_key_values = DynamicCache()
```

### 4. **Memory and Compute Optimizations**
```python
# Enable mixed precision for T3 model
from torch.cuda.amp import autocast

with autocast():
    # T3 inference calls
```

### 5. **Batch Processing for S3Gen**
The S3Gen model could potentially process multiple token chunks in parallel.

### 6. **Attention Implementation Override**
Force specific attention implementation during model loading:
```python
import transformers
transformers.modeling_utils.PreTrainedModel._attn_implementation = "flash_attention_2"
```

## Performance vs Quality Trade-offs

| Configuration | RTF Expected | Audio Quality | Latency |
|---------------|--------------|---------------|---------|
| chunk_size=50, context=100 | ~0.9 | Excellent | Medium |
| chunk_size=75, context=150 | ~0.7 | Excellent | Medium |
| chunk_size=100, context=200 | ~0.6 | Excellent | Higher |
| chunk_size=75, context=75 | ~0.6 | Good | Medium |

## Next Steps Priority

1. **Add model compilation** - Likely 20-30% speedup
2. **Fix KV cache format** - Remove overhead from deprecated warnings
3. **Test Flash Attention 2** - If available, significant speedup
4. **Profile S3Gen bottlenecks** - May be the limiting factor now
5. **Consider mixed precision** - Memory and speed benefits

## Current Status
- ✅ Disabled alignment stream analyzer (major bottleneck removed)
- ✅ Optimized chunk sizes
- ⚠️ Context window needs to stay ≥50 for quality
- 🔄 Model compilation and cache optimization are next big wins
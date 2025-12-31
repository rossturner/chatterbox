# Fix CUDA Graph Invalidation on StaticCache Resize

## Problem Statement

Chatterbox TTS experiences intermittent freezes (complete hang) and extreme slowdowns (70 it/s → 5 it/s) during generation. These failures are non-deterministic and occur mid-generation, typically around 50-60% progress.

## Root Cause Analysis

### Observed Behavior

1. **Complete Freeze**: Generation stops at exactly the same iteration (e.g., 540/1000) with no progress for 300+ seconds until timeout
2. **Extreme Slowdown**: Generation speed drops from ~70 it/s to ~1-5 it/s, causing 300s timeout

### Investigation Findings

Both failure cases showed a common pattern in the logs:

```
WARNING - Total sequence length 1494 exceeds max_cache_len 1200. Adjusting.
...
Sampling:  54%|█████▍| 540/1000 [00:09<00:06, 73.88it/s]
Sampling:  54%|█████▍| 540/1000 [00:22<00:06, 73.88it/s]  # FROZEN - same iteration 13s later
...
TimeoutError: Inference task 'generate_zh_99' timed out after 300.0s
```

### Root Cause

**CUDA graphs capture fixed memory addresses during compilation.** When `torch.compile(mode='max-autotune')` compiles the model, it creates CUDA graphs that reference specific GPU memory locations.

The current code in `src/chatterbox/models/t3/t3.py` dynamically resizes the `StaticCache` when sequence length exceeds `max_cache_len`:

```python
# Lines 377-379
if total_len > max_cache_len:
    logger.warning(f"Total sequence length {total_len} exceeds max_cache_len {max_cache_len}. Adjusting.")
    max_cache_len = total_len

# Lines 381-388 - recreates cache with new size
kv_cache = self.get_cache(...)
```

When the cache is recreated with a larger size:
1. Old `StaticCache` tensors are deallocated
2. New `StaticCache` tensors are allocated at different memory addresses
3. The compiled CUDA graph still references the old memory addresses
4. GPU operations hang waiting for data that will never arrive (freeze) or fall back to slow unoptimized path (slowdown)

## Data Analysis

From production logs (7216 successful generations):

| Sequence Length | Count | Percentage |
|----------------|-------|------------|
| ≤1200 | ~5432 | 75.3% |
| 1201-1400 | 1307 | 18.1% |
| 1401-1600 | 371 | 5.1% |
| 1601-1800 | 92 | 1.3% |
| 1801-2000 | 12 | 0.17% |
| 2000+ | 2 | 0.03% |

- **Max observed sequence length**: 2091
- **Requests triggering cache resize**: 1784 (24.7%)

## Proposed Fix

### Option 1: Increase `REDUCED_CACHE_LEN` (Recommended)

In `src/server/optimized_model_loader.py`:

```python
# Current (line 36)
REDUCED_CACHE_LEN = 1200  # Reduced from 4096 for better performance

# Proposed
REDUCED_CACHE_LEN = 2200  # Covers max observed (2091) + buffer
```

**Pros:**
- Simple change
- Prevents cache resizing in 100% of observed cases
- Maintains CUDA graph optimization benefits

**Cons:**
- Slightly higher VRAM usage (~1.8x increase in KV cache memory)
- May need adjustment if longer sequences are encountered

### Option 2: Restore Original Cache Size

```python
REDUCED_CACHE_LEN = 4096  # Original value
```

**Pros:**
- Maximum safety margin
- No future adjustments needed

**Cons:**
- Higher VRAM usage (~3.4x increase in KV cache memory)
- May impact performance on memory-constrained systems

### Option 3: Disable CUDA Graphs

In `src/server/optimized_model_loader.py`:

```python
# Current (line 37)
COMPILE_MODE = "max-autotune"  # Uses CUDA graphs

# Proposed
COMPILE_MODE = "reduce-overhead"  # No CUDA graphs
```

**Pros:**
- Eliminates the CUDA graph invalidation issue entirely
- Cache can be dynamically resized safely

**Cons:**
- ~30-50% performance degradation (lose CUDA graph benefits)
- Not recommended unless memory is severely constrained

## Recommended Implementation

1. Set `REDUCED_CACHE_LEN = 2200` to cover current workload
2. Add monitoring for max sequence length to detect if further adjustment needed
3. Consider making `REDUCED_CACHE_LEN` configurable via environment variable for easier tuning

## Verification

After implementing the fix:
1. Run extended workload and verify no "exceeds max_cache_len" warnings appear
2. Monitor for freezes/timeouts in generation
3. Check VRAM usage stays within acceptable bounds

## References

- `src/server/optimized_model_loader.py`: CUDA graph compilation configuration
- `src/chatterbox/models/t3/t3.py`: StaticCache resize logic (lines 377-388)
- `src/server/inference_thread.py`: 300s timeout configuration

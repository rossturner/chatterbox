# Advanced Performance Optimization Plan

## Overview
This plan implements model compilation and KV cache modernization as **separate optimization layers** that wrap the existing models without modifying core implementation.

## Architecture Design

### 1. **Performance Wrapper Pattern**
```
ChatterboxTTS (unchanged)
    ↓
OptimizedChatterboxTTS (new wrapper)
    ↓ 
- CompiledModelWrapper
- ModernKVCacheManager  
- FlashAttentionOptimizer
```

### 2. **Implementation Strategy**

#### A. **Compiled Model Wrapper**
- Wraps existing models with torch.compile
- Selective compilation (only performance-critical parts)
- Fallback to original models if compilation fails
- Compilation mode selection (default, reduce-overhead, max-autotune)

#### B. **Modern KV Cache Manager**
- Replaces deprecated tuple format with DynamicCache
- Manages cache lifecycle and memory optimization
- Transparent to existing generation logic
- Automatic format conversion

#### C. **Flash Attention Optimizer**
- Detects and enables Flash Attention 2 if available
- Forces optimal attention implementation
- Handles attention mask optimization

## File Structure

```
src/chatterbox/
├── optimizations/              # New optimization module
│   ├── __init__.py
│   ├── compiled_wrapper.py     # Model compilation wrapper
│   ├── kv_cache_modern.py      # Modern KV cache implementation
│   ├── attention_optimizer.py  # Attention optimizations
│   └── optimized_tts.py        # Main optimized wrapper
├── tts.py                      # Original (unchanged)
└── models/                     # Original models (unchanged)
```

## Implementation Plan

### Phase 1: Foundation (Day 1)
1. Create optimization module structure
2. Implement base OptimizedChatterboxTTS wrapper
3. Add configuration system for enabling/disabling optimizations

### Phase 2: Model Compilation (Day 1-2)
1. Implement CompiledModelWrapper
2. Add selective compilation for T3 and S3Gen
3. Implement compilation failure fallbacks
4. Add compilation mode selection

### Phase 3: KV Cache Modernization (Day 2)
1. Implement ModernKVCacheManager
2. Add DynamicCache integration
3. Create transparent cache format conversion
4. Handle cache lifecycle management

### Phase 4: Attention Optimization (Day 2-3)
1. Implement FlashAttentionOptimizer
2. Add Flash Attention 2 detection/enablement
3. Optimize attention mask handling
4. Force optimal attention implementations

### Phase 5: Integration & Testing (Day 3)
1. Integrate all optimizations into OptimizedChatterboxTTS
2. Add comprehensive benchmarking
3. Performance regression testing
4. Documentation and usage examples

## Technical Details

### Model Compilation Strategy
```python
# Selective compilation - only performance-critical components
class CompiledModelWrapper:
    def __init__(self, model, compile_config):
        self.original_model = model
        self.compiled_components = {}
        
        # Compile only bottleneck components
        if compile_config.get('t3_transformer', True):
            self.compiled_components['tfmr'] = torch.compile(
                model.t3.tfmr, 
                mode=compile_config.get('mode', 'reduce-overhead')
            )
        
        if compile_config.get('s3gen', True):
            self.compiled_components['s3gen'] = torch.compile(
                model.s3gen,
                mode=compile_config.get('mode', 'reduce-overhead')
            )
```

### KV Cache Modernization
```python
from transformers import DynamicCache

class ModernKVCacheManager:
    def __init__(self):
        self.cache = DynamicCache()
    
    def convert_legacy_cache(self, past_key_values):
        """Convert tuple format to DynamicCache"""
        if isinstance(past_key_values, tuple):
            # Convert tuple of tuples to modern format
            return self._tuple_to_dynamic_cache(past_key_values)
        return past_key_values
    
    def optimize_cache_memory(self):
        """Optimize cache memory usage"""
        self.cache.crop(-1)  # Remove oldest entries if needed
```

### Usage Interface
```python
# Drop-in replacement for ChatterboxTTS
from chatterbox.optimizations import OptimizedChatterboxTTS

# Enable all optimizations
model = OptimizedChatterboxTTS.from_pretrained(
    device="cuda",
    optimizations={
        'compile_models': True,
        'compile_mode': 'reduce-overhead',
        'modern_kv_cache': True,
        'flash_attention': True,
        'selective_compilation': {
            't3_transformer': True,
            's3gen': True,
            'voice_encoder': False  # Skip if not bottleneck
        }
    }
)

# Same API as original
for audio_chunk, metrics in model.generate_stream(...):
    # Process audio
```

## Benefits

### 1. **Non-Invasive**
- Original models remain unchanged
- Easy to enable/disable optimizations
- No risk to existing functionality

### 2. **Modular**
- Each optimization can be enabled independently
- Easy to add new optimizations
- Clear separation of concerns

### 3. **Fallback Safe**
- Automatic fallback to original implementation
- Graceful degradation if optimizations fail
- Development/production mode switching

### 4. **Performance Focused**
- Target 20-40% additional speedup
- Memory optimization
- Reduced overhead

## Expected Performance Gains

| Optimization | Expected RTF Improvement |
|--------------|-------------------------|
| Model Compilation | 20-30% |
| Modern KV Cache | 5-10% |
| Flash Attention 2 | 10-20% |
| **Combined** | **30-50%** |

**Target:** Current RTF ~0.8 → Optimized RTF ~0.4-0.6

## Risk Mitigation

1. **Compilation Failures:** Automatic fallback to original models
2. **Memory Issues:** Progressive optimization with monitoring
3. **Compatibility:** Extensive testing across PyTorch versions
4. **Debugging:** Optional verbose logging and profiling modes

This approach gives you the performance benefits while maintaining the stability and integrity of your core implementation.
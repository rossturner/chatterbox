# Mixed Precision Optimization Plan for Chatterbox TTS

## Executive Summary

This document outlines a comprehensive mixed precision optimization strategy for the Chatterbox TTS system, targeting FP16/BF16 precision with TF32 matmul acceleration on RTX 4090's Tensor Cores. Analysis indicates potential for **~1.3x speed improvement and 50% VRAM reduction** while maintaining audio quality.

## Current State Analysis

### Existing Precision Configuration

**Current Setup:**
- **T3 Model (Llama Backbone)**: Uses `torch_dtype="bfloat16"` in configuration (llama_configs.py)
- **S3Gen Vocoder**: Mixed precision already partially implemented
  - Flow module: Uses `.half()` conversions for `prompt_feat` and `embedding`
  - Decoder supports dtype checking: `torch.float32, torch.bfloat16, torch.float16`
- **Voice Encoder**: Full FP32 precision (LSTM + projection layers)
- **Model Loading**: No automatic precision casting during model initialization

**Quantized Model Variants Already Available:**
- `quantized_models/float16_weights/`: FP16 weight quantization (~1.5GB, 50% size reduction)
- `quantized_models/mixed_precision/`: Mixed precision implementation (~1.5GB)

**Performance Baseline (from test harness):**
```
Original GRPO Model: 3045 MB, Peak VRAM: 4.97 GB, RTF: 0.77
Float16 Weights:     1523 MB, Peak VRAM: 3.62 GB, RTF: 0.74  
Mixed Precision:     1523 MB, Peak VRAM: 3.61 GB, RTF: 0.76
```

### TF32 Support Status

**Current Configuration:**
- TF32 not explicitly enabled in codebase
- PyTorch default: TF32 enabled for convolutions, disabled for matmul
- RTX 4090 has 512 4th-gen Tensor Cores supporting TF32/FP16/BF16

## Precision Format Analysis

### Format Comparison for TTS Models

| Format | Range | Precision | TTS Suitability | Memory | Speed |
|--------|-------|-----------|----------------|---------|-------|
| **FP32** | ±3.4×10³⁸ | 23 bits | Baseline quality | 4 bytes | 1.0x |
| **BF16** | ±3.4×10³⁸ | 7 bits | Excellent stability | 2 bytes | 1.3-1.9x |
| **FP16** | ±65,504 | 10 bits | Good with care | 2 bytes | 1.3-1.8x |
| **TF32** | ±3.4×10³⁸ | 10 bits | Excellent for matmul | 4 bytes* | 1.3x |

*TF32 uses FP32 storage but TF32 computation

### TTS-Specific Considerations

**Research Findings:**
- NVIDIA Tacotron2 benchmarks show **TF32 can outperform FP16** for TTS models
- BF16 provides better numerical stability than FP16 for audio generation
- Voice encoder (LSTM) benefits significantly from half-precision
- Vocoder quality-sensitive operations should maintain higher precision

## Implementation Strategy

### Phase 1: TF32 Matmul Acceleration (Low Risk)

**Implementation:**
```python
# Enable TF32 globally at startup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  # Already enabled by default
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
```

**Target Components:**
- T3 Llama transformer matmul operations
- Attention mechanism computations
- Linear projection layers

**Expected Impact:**
- Speed: +30% for matmul-heavy operations
- VRAM: No change (uses FP32 storage)
- Quality: Negligible impact (19-bit mantissa vs 23-bit FP32)

### Phase 2: Model-Specific Mixed Precision

**T3 Model (Text-to-Speech-Tokens):**
```python
class T3(nn.Module):
    def __init__(self, hp=T3Config()):
        super().__init__()
        # ... existing init ...
        
        # Enable mixed precision support
        self.mixed_precision_enabled = False
        self.precision_dtype = torch.float32
    
    def enable_mixed_precision(self, dtype=torch.bfloat16):
        """Enable mixed precision for T3 model"""
        self.precision_dtype = dtype
        self.mixed_precision_enabled = True
        
        # Convert embeddings to half precision
        self.text_emb = self.text_emb.to(dtype=dtype)
        self.speech_emb = self.speech_emb.to(dtype=dtype)
        
        # Convert Llama backbone
        self.tfmr = self.tfmr.to(dtype=dtype)
        
        # Keep projection heads in FP32 for stability
        # self.text_head remains FP32
        # self.speech_head remains FP32
```

**S3Gen Vocoder (Enhanced):**
```python
class S3Token2Wav(S3Token2Mel):
    def enable_mixed_precision(self, dtype=torch.bfloat16):
        """Enable mixed precision for vocoder"""
        # Flow matching operations in half precision
        self.flow = self.flow.to(dtype=dtype)
        
        # Keep mel extraction and final audio generation in FP32
        # for quality preservation
        
        # Enable autocast for forward pass
        self.use_autocast = True
        self.autocast_dtype = dtype
```

**Voice Encoder (Most Beneficial):**
```python
class VoiceEncoder(nn.Module):
    def enable_mixed_precision(self, dtype=torch.float16):
        """Voice encoder benefits most from FP16"""
        self.lstm = self.lstm.to(dtype=dtype)
        self.proj = self.proj.to(dtype=dtype)
        self.precision_dtype = dtype
```

### Phase 3: Autocast Integration

**ChatterboxTTS Integration:**
```python
class ChatterboxTTS:
    def __init__(self, ...):
        # ... existing init ...
        self.mixed_precision_config = {
            'enabled': False,
            'dtype': torch.bfloat16,
            'use_autocast': True,
            'tf32_enabled': False
        }
    
    def enable_mixed_precision(self, 
                             dtype=torch.bfloat16, 
                             enable_tf32=True,
                             components=['t3', 's3gen', 've']):
        """Enable mixed precision across specified components"""
        
        if enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            self.mixed_precision_config['tf32_enabled'] = True
        
        self.mixed_precision_config.update({
            'enabled': True,
            'dtype': dtype,
            'components': components
        })
        
        # Component-specific precision
        if 't3' in components:
            self.t3.enable_mixed_precision(dtype=dtype)
        if 's3gen' in components:
            self.s3gen.enable_mixed_precision(dtype=dtype)  
        if 've' in components:
            # Voice encoder works well with FP16
            ve_dtype = torch.float16 if dtype == torch.bfloat16 else dtype
            self.ve.enable_mixed_precision(dtype=ve_dtype)
    
    def generate(self, text, **kwargs):
        """Generate with mixed precision support"""
        if not self.mixed_precision_config['enabled']:
            return self._generate_fp32(text, **kwargs)
        
        dtype = self.mixed_precision_config['dtype']
        
        with torch.autocast(device_type='cuda', dtype=dtype, enabled=True):
            return self._generate_mixed_precision(text, **kwargs)
```

## Quality Preservation Strategies

### Precision-Sensitive Operations (Keep FP32)

1. **Final Audio Generation:**
   - Mel-to-waveform conversion
   - Audio watermarking
   - Sample rate conversion

2. **Loss-Sensitive Computations:**
   - Final linear projection heads
   - Loss computations during training
   - Gradient accumulation

3. **Numerical Stability Critical:**
   - LayerNorm operations
   - Softmax with extreme values
   - Division operations

### Autocast Exclusions

```python
# Example implementation
def inference_with_precision_control(self, ...):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # Most operations in BF16
        hidden_states = self.transformer_layers(input_embeds)
        
        # Critical operations in FP32
        with torch.autocast(device_type='cuda', enabled=False):
            logits = self.output_projection(hidden_states.float())
            
        return logits
```

## Performance Benchmarking Approach

### Test Configuration

**Hardware Target:** RTX 4090 (Ada Lovelace, 512 Tensor Cores)

**Test Cases:**
```python
precision_configs = [
    {'name': 'Baseline FP32', 'dtype': torch.float32, 'tf32': False},
    {'name': 'TF32 Only', 'dtype': torch.float32, 'tf32': True},
    {'name': 'BF16 + TF32', 'dtype': torch.bfloat16, 'tf32': True},
    {'name': 'FP16 + TF32', 'dtype': torch.float16, 'tf32': True},
    {'name': 'Selective Mixed', 'dtype': 'mixed', 'tf32': True},
]
```

**Metrics to Track:**
- Generation time per token
- VRAM usage (peak and steady-state)
- Real-Time Factor (RTF)
- Audio quality metrics (MOS via automated evaluation)
- Memory bandwidth utilization

### Enhanced Performance Test Harness

```python
# Extension to existing performance_test_harness.py
class MixedPrecisionTestConfig:
    def __init__(self, name, dtype, enable_tf32=True, components=None):
        self.name = name
        self.dtype = dtype
        self.enable_tf32 = enable_tf32
        self.components = components or ['t3', 's3gen', 've']

def run_precision_benchmarks():
    configs = [
        MixedPrecisionTestConfig("Baseline FP32", torch.float32, False),
        MixedPrecisionTestConfig("TF32 Enabled", torch.float32, True),
        MixedPrecisionTestConfig("BF16 + TF32", torch.bfloat16, True),
        MixedPrecisionTestConfig("FP16 + TF32", torch.float16, True),
        MixedPrecisionTestConfig("VE Only FP16", torch.float32, True, ['ve']),
    ]
    
    for config in configs:
        model = ChatterboxTTS.from_pretrained(device)
        model.enable_mixed_precision(
            dtype=config.dtype,
            enable_tf32=config.enable_tf32,
            components=config.components
        )
        # Run existing test harness...
```

## Risk Assessment & Mitigation

### High Risk Areas

**1. Gradient Overflow (FP16)**
- **Risk:** NaN gradients during training
- **Mitigation:** Use GradScaler, implement gradient clipping
- **Fallback:** Switch to BF16 for problematic components

**2. Audio Quality Degradation**
- **Risk:** Reduced precision affecting mel spectrogram quality
- **Mitigation:** Keep vocoder final stages in FP32
- **Testing:** A/B comparison with reference audio

**3. Numerical Instability**
- **Risk:** LSTM hidden states, attention scores overflow
- **Mitigation:** Selective FP32 for critical operations
- **Monitoring:** Loss curve stability, NaN detection

### Medium Risk Areas

**1. Model Loading Compatibility**
- **Risk:** Existing checkpoints incompatible with new precision
- **Mitigation:** Automatic conversion utilities, version checking

**2. Performance Regression**
- **Risk:** Some operations slower in mixed precision
- **Mitigation:** Component-level benchmarking, selective application

### Low Risk Areas

**1. TF32 Integration**
- **Risk:** Minimal quality impact expected
- **Benefit:** Pure performance gain

**2. Voice Encoder Precision**
- **Risk:** LSTM well-suited for FP16
- **Benefit:** Significant VRAM reduction

## Implementation Timeline

### Week 1: Foundation
- [ ] Enable TF32 globally
- [ ] Basic mixed precision infrastructure
- [ ] Component-level precision control

### Week 2: Integration  
- [ ] T3 model mixed precision
- [ ] S3Gen selective precision
- [ ] Voice encoder FP16 conversion

### Week 3: Testing & Tuning
- [ ] Performance benchmarking
- [ ] Quality validation
- [ ] Memory usage optimization

### Week 4: Production Ready
- [ ] Error handling & fallbacks
- [ ] Documentation updates
- [ ] User configuration options

## Expected Outcomes

### Performance Targets

**Speed Improvements:**
- T3 inference: +25-35% (matmul-heavy operations)
- Voice encoding: +40-50% (LSTM acceleration) 
- S3Gen vocoder: +15-25% (selective mixed precision)
- Overall RTF: 0.76 → 0.55-0.60 (20-25% improvement)

**Memory Efficiency:**
- Model size: 50% reduction for affected components
- Peak VRAM: 4.97GB → 3.0-3.5GB (25-30% reduction)
- Memory bandwidth: +40% effective utilization

**Quality Preservation:**
- Target: <2% MOS degradation
- Critical: No audio artifacts or watermark issues
- Benchmark: A/B testing against FP32 baseline

## Rollback Plan

### Immediate Rollback Triggers
- Audio quality degradation >5% MOS
- Generation failures >1% 
- VRAM usage increase
- Performance regression >10%

### Rollback Procedure
```python
# Emergency fallback to FP32
def disable_mixed_precision(model):
    model.t3 = model.t3.float()
    model.s3gen = model.s3gen.float()  
    model.ve = model.ve.float()
    torch.backends.cuda.matmul.allow_tf32 = False
    return model
```

### Component-Level Rollback
- Selective disable: Keep working components in mixed precision
- Gradual rollback: TF32 → BF16 → FP16 → FP32
- Configuration-based: Runtime precision switching

## Monitoring & Validation

### Automated Quality Checks
```python
def validate_mixed_precision_quality(model, test_cases):
    quality_metrics = {
        'similarity_score': [],
        'spectral_distortion': [],
        'perceptual_quality': []
    }
    
    for audio_ref, transcript in test_cases:
        fp32_output = model.generate(transcript)  # Baseline
        
        model.enable_mixed_precision()
        mixed_output = model.generate(transcript)  # Test
        
        quality_metrics['similarity_score'].append(
            compute_similarity(fp32_output, mixed_output)
        )
        
    return quality_metrics
```

### Performance Monitoring
- Real-time RTF tracking
- VRAM usage alerts
- Generation failure detection
- Quality metric logging

## Conclusion

Mixed precision optimization represents a high-impact, moderate-risk enhancement for Chatterbox TTS. The combination of TF32 matmul acceleration and selective BF16/FP16 application should deliver significant performance improvements while maintaining audio quality.

**Key Success Factors:**
1. **Gradual rollout** with component-level precision control
2. **Comprehensive testing** across diverse audio content
3. **Quality-first approach** with automatic fallbacks
4. **Performance validation** on target RTX 4090 hardware

**Immediate Next Steps:**
1. Enable TF32 for immediate low-risk performance gain
2. Implement Voice Encoder FP16 for maximum VRAM benefit
3. Establish benchmarking framework for validation

This optimization aligns with the existing quantized model infrastructure and builds upon the current performance test harness, ensuring a smooth integration path while delivering substantial improvements to the Chatterbox TTS streaming system.
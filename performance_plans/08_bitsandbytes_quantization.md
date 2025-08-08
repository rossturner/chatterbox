# Bitsandbytes Quantization Optimization Plan for Chatterbox TTS

## Executive Summary

This document outlines a comprehensive bitsandbytes quantization strategy for the Chatterbox TTS system, targeting **50-70% memory reduction** through NF4 4-bit and INT8 8-bit quantization while maintaining audio quality. Analysis shows potential for substantial VRAM savings on RTX 4090 with minimal quality degradation, particularly beneficial for the 520M-parameter Llama backbone.

## Current State Analysis

### Existing Quantization Infrastructure

**Current Quantized Variants:**
- **Float16 Weights**: `quantized_models/float16_weights/` (~1.5GB, 50% size reduction)
- **Mixed Precision**: `quantized_models/mixed_precision/` (~1.5GB, similar performance)
- **Performance Baseline**: RTX 4090 peak VRAM 3.6-4.97GB, RTF 0.74-0.77

**Model Components Analysis:**
- **T3 Model (520M Llama)**: Primary quantization target (~2GB in FP32)
  - 30 transformer layers with 16 attention heads
  - Hidden size: 1024, Intermediate: 4096
  - Already configured with `torch_dtype="bfloat16"`
- **S3Gen Vocoder**: Secondary target (~800MB)
- **Voice Encoder**: Smaller footprint (~200MB) but good quantization candidate
- **S3Tokenizer**: Minimal impact (~50MB)

**Dependencies:**
- **Bitsandbytes**: v0.46.1 already installed
- **Transformers**: v4.46.3 with quantization support
- **PyTorch**: v2.6.0 with CUDA 12.4 compatibility

### Memory Usage Patterns

**Current VRAM Distribution (RTX 4090):**
```
Component         FP32    FP16    Current Peak
T3 Llama Backbone  ~2.0GB  ~1.0GB   ~1.5GB
S3Gen Vocoder      ~800MB  ~400MB   ~600MB
Voice Encoder      ~200MB  ~100MB   ~150MB
Activations        ~1.0GB  ~800MB   ~1.2GB
Overhead           ~300MB  ~300MB   ~400MB
Total             ~4.3GB  ~2.6GB   ~3.85GB
```

## Quantization Strategy Analysis

### NF4 vs INT8 Trade-offs for TTS Models

| Aspect | NF4 4-bit | INT8 8-bit | FP16 Baseline |
|--------|-----------|------------|---------------|
| **Memory Reduction** | 75% | 50% | 50% |
| **Model Size** | ~380MB | ~760MB | ~1.5GB |
| **Quality Impact** | Low-Medium | Very Low | Baseline |
| **Inference Speed** | 0.9-1.1x | 1.0-1.2x | 1.0x |
| **Training Support** | QLoRA | Full | Full |

### Component-Specific Quantization Suitability

**T3 Llama Backbone (High Priority):**
- **NF4 Recommendation**: Excellent candidate for 4-bit quantization
- **Reasoning**: Transformer weights typically normally distributed
- **Expected Savings**: 2GB → 500MB (75% reduction)
- **Quality Impact**: Minimal for TTS (less sensitive than pure text generation)

**S3Gen Vocoder (Medium Priority):**
- **INT8 Recommendation**: 8-bit preferred for audio quality preservation
- **Reasoning**: Flow-matching operations sensitive to precision
- **Expected Savings**: 800MB → 400MB (50% reduction)
- **Quality Impact**: Low with proper compute dtype selection

**Voice Encoder (High ROI):**
- **NF4 Recommendation**: Excellent ROI despite smaller size
- **Reasoning**: LSTM + projection layers quantize well
- **Expected Savings**: 200MB → 50MB (75% reduction)
- **Quality Impact**: Negligible (embedding extraction robust)

## Implementation Plan

### Phase 1: NF4 Quantization Infrastructure

#### Core Quantization Configuration

```python
from transformers import BitsAndBytesConfig
import torch

class ChatterboxQuantizationConfig:
    """Centralized quantization configuration for Chatterbox components"""
    
    @staticmethod
    def get_nf4_config(compute_dtype=torch.bfloat16, double_quant=True):
        """NF4 4-bit configuration optimized for TTS models"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=double_quant,  # Additional 0.4 bits/param savings
            bnb_4bit_compute_dtype=compute_dtype,    # BF16 for stability
            bnb_4bit_quant_storage=torch.uint8,     # Storage optimization
        )
    
    @staticmethod
    def get_int8_config():
        """INT8 8-bit configuration for quality-sensitive components"""
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,                 # Mixed precision threshold
            llm_int8_enable_fp32_cpu_offload=False, # Keep on GPU
        )
    
    @staticmethod
    def get_mixed_config(t3_4bit=True, s3gen_8bit=True, ve_4bit=True):
        """Component-specific mixed quantization strategy"""
        return {
            't3': ChatterboxQuantizationConfig.get_nf4_config() if t3_4bit else None,
            's3gen': ChatterboxQuantizationConfig.get_int8_config() if s3gen_8bit else None,
            've': ChatterboxQuantizationConfig.get_nf4_config() if ve_4bit else None,
        }
```

#### Enhanced ChatterboxTTS Integration

```python
class ChatterboxTTS:
    def __init__(self, ...):
        # ... existing init ...
        self.quantization_config = None
        self.is_quantized = False
    
    @classmethod
    def from_pretrained_quantized(
        cls, 
        device, 
        quantization_strategy='mixed',
        force_4bit=False
    ) -> 'ChatterboxTTS':
        """Load Chatterbox with bitsandbytes quantization"""
        
        if quantization_strategy == 'nf4':
            quant_config = ChatterboxQuantizationConfig.get_nf4_config()
        elif quantization_strategy == 'int8':
            quant_config = ChatterboxQuantizationConfig.get_int8_config()
        elif quantization_strategy == 'mixed':
            quant_config = ChatterboxQuantizationConfig.get_mixed_config()
        else:
            raise ValueError(f"Unknown quantization strategy: {quantization_strategy}")
        
        # Component-specific quantized loading
        instance = cls._load_with_quantization(device, quant_config)
        instance.quantization_config = quant_config
        instance.is_quantized = True
        
        return instance
    
    @classmethod
    def _load_with_quantization(cls, device, quant_config):
        """Internal quantized loading with component-specific handling"""
        
        # T3 Model - NF4 quantization for Llama backbone
        if isinstance(quant_config, dict) and 't3' in quant_config:
            t3_config = quant_config['t3']
            # Apply quantization to LlamaModel within T3
            t3 = T3()
            t3.tfmr = LlamaModel.from_pretrained(
                "path/to/llama/weights",
                quantization_config=t3_config,
                torch_dtype=torch.bfloat16,
                device_map={"": device}
            )
        
        # S3Gen - INT8 quantization for vocoder stability
        if isinstance(quant_config, dict) and 's3gen' in quant_config:
            s3gen_config = quant_config['s3gen']
            s3gen = S3Gen()
            # Apply quantization to flow and decoder components
            s3gen = quantize_model_selective(s3gen, s3gen_config)
        
        # Voice Encoder - NF4 for maximum memory savings
        if isinstance(quant_config, dict) and 've' in quant_config:
            ve_config = quant_config['ve']
            ve = VoiceEncoder()
            ve = quantize_model(ve, ve_config)
        
        return cls(t3, s3gen, ve, tokenizer, device)
```

### Phase 2: Component-Specific Quantization

#### T3 Model Quantization (Primary Target)

```python
class T3(nn.Module):
    def enable_bitsandbytes_quantization(self, config_type='nf4'):
        """Enable bitsandbytes quantization for T3 components"""
        
        if config_type == 'nf4':
            quant_config = ChatterboxQuantizationConfig.get_nf4_config()
        elif config_type == 'int8':
            quant_config = ChatterboxQuantizationConfig.get_int8_config()
        
        # Quantize the Llama transformer backbone
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        
        if config_type == 'nf4':
            self._replace_linear_layers_4bit(self.tfmr, quant_config)
        else:
            self._replace_linear_layers_8bit(self.tfmr, quant_config)
        
        # Keep embedding layers and heads in higher precision
        # for stability (small memory impact, large quality benefit)
        self.quantization_enabled = True
        self.quantization_config = quant_config
    
    def _replace_linear_layers_4bit(self, module, config):
        """Replace nn.Linear with Linear4bit recursively"""
        from bitsandbytes.nn import Linear4bit
        
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Replace with 4-bit linear layer
                quant_layer = Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=config.bnb_4bit_compute_dtype,
                    quant_type=config.bnb_4bit_quant_type,
                    use_double_quant=config.bnb_4bit_use_double_quant
                )
                setattr(module, name, quant_layer)
            else:
                self._replace_linear_layers_4bit(child, config)
```

#### S3Gen Vocoder Quantization (Quality-Sensitive)

```python
class S3Gen(nn.Module):
    def enable_selective_quantization(self, quantize_flow=True, quantize_decoder=False):
        """Selective quantization for S3Gen preserving audio quality"""
        
        if quantize_flow:
            # Flow matching layers can handle 8-bit quantization well
            int8_config = ChatterboxQuantizationConfig.get_int8_config()
            self.flow = quantize_module_int8(self.flow, int8_config)
        
        if quantize_decoder:
            # Decoder quantization more aggressive but risky for quality
            nf4_config = ChatterboxQuantizationConfig.get_nf4_config()
            self.decoder = quantize_module_nf4(self.decoder, nf4_config)
        
        # Always keep final audio generation layers in FP16/BF16
        # for maximum quality preservation
        self.hift.requires_grad_(False)  # Freeze for stability
```

#### Voice Encoder Quantization (High ROI)

```python
class VoiceEncoder(nn.Module):
    def enable_aggressive_quantization(self):
        """Aggressive quantization for voice encoder (embedding extraction robust)"""
        
        nf4_config = ChatterboxQuantizationConfig.get_nf4_config(
            double_quant=True  # Maximum compression for embeddings
        )
        
        # Quantize LSTM and projection layers
        self.lstm = quantize_lstm_4bit(self.lstm, nf4_config)
        self.proj = quantize_linear_4bit(self.proj, nf4_config)
        
        self.is_quantized = True
```

### Phase 3: Advanced Quantization Features

#### Double Quantization Implementation

```python
class AdvancedQuantizationManager:
    """Advanced quantization features for maximum memory efficiency"""
    
    @staticmethod
    def enable_double_quantization():
        """Enable nested quantization for additional 0.4 bits/param savings"""
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # Key feature: quantize quantization constants
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8
        )
        return config
    
    @staticmethod
    def optimize_memory_layout():
        """Optimize memory layout for quantized models"""
        # Enable memory efficient attention
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # Configure optimal memory format
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        return True
```

#### Dynamic Quantization Strategy

```python
class DynamicQuantizationStrategy:
    """Runtime quantization strategy selection based on available VRAM"""
    
    def __init__(self, device):
        self.device = device
        self.vram_threshold_4bit = 8.0  # GB
        self.vram_threshold_8bit = 12.0  # GB
    
    def select_optimal_strategy(self):
        """Select quantization strategy based on available VRAM"""
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            if vram_gb < self.vram_threshold_4bit:
                return 'aggressive_nf4'  # Maximum compression
            elif vram_gb < self.vram_threshold_8bit:
                return 'mixed'           # Balanced approach
            else:
                return 'conservative'    # Quality-first
        
        return 'int8'  # Safe default
```

## Performance Benchmarking Framework

### Enhanced Test Harness Integration

```python
# Extension to performance_test_harness.py

class QuantizationBenchmarkConfig:
    """Benchmark configuration for quantization evaluation"""
    
    def __init__(self, name, quantization_strategy, expected_vram_reduction):
        self.name = name
        self.quantization_strategy = quantization_strategy
        self.expected_vram_reduction = expected_vram_reduction

def run_quantization_benchmarks():
    """Comprehensive quantization performance evaluation"""
    
    configs = [
        # Baseline
        QuantizationBenchmarkConfig("FP16 Baseline", None, 0.0),
        
        # Conservative quantization
        QuantizationBenchmarkConfig("INT8 Conservative", "int8", 0.25),
        
        # Balanced approaches
        QuantizationBenchmarkConfig("Mixed Quant", "mixed", 0.45),
        QuantizationBenchmarkConfig("NF4 Standard", "nf4", 0.60),
        
        # Aggressive compression
        QuantizationBenchmarkConfig("NF4 Double Quant", "nf4_double", 0.70),
        QuantizationBenchmarkConfig("NF4 Aggressive", "nf4_aggressive", 0.75),
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting {config.name}...")
        
        # Load model with quantization
        if config.quantization_strategy:
            model = ChatterboxTTS.from_pretrained_quantized(
                device="cuda",
                quantization_strategy=config.quantization_strategy
            )
        else:
            model = ChatterboxTTS.from_pretrained("cuda")
        
        # Run existing test harness with quantized model
        test_results = run_performance_test_with_model(model, config)
        results.append(test_results)
        
        # Cleanup for accurate memory measurement
        del model
        torch.cuda.empty_cache()
    
    return analyze_quantization_results(results)
```

### Quality Assessment Framework

```python
class QuantizationQualityAssessment:
    """Automated quality assessment for quantized models"""
    
    def __init__(self, reference_model, test_cases):
        self.reference_model = reference_model
        self.test_cases = test_cases
    
    def evaluate_quantized_model(self, quantized_model):
        """Comprehensive quality evaluation"""
        metrics = {
            'spectral_distortion': [],
            'mel_distance': [],
            'perceptual_similarity': [],
            'speaker_similarity': []
        }
        
        for audio_ref, transcript in self.test_cases:
            # Generate with reference and quantized models
            ref_audio = self.reference_model.generate(transcript)
            quant_audio = quantized_model.generate(transcript)
            
            # Compute quality metrics
            metrics['spectral_distortion'].append(
                self._compute_spectral_distortion(ref_audio, quant_audio)
            )
            
            metrics['mel_distance'].append(
                self._compute_mel_distance(ref_audio, quant_audio)
            )
            
            # Additional TTS-specific metrics
            metrics['perceptual_similarity'].append(
                self._compute_perceptual_similarity(ref_audio, quant_audio)
            )
            
        return self._aggregate_metrics(metrics)
    
    def _compute_spectral_distortion(self, ref_audio, test_audio):
        """Compute spectral distortion between audio samples"""
        # Implementation of spectral distance metrics
        # Returns score where lower is better
        pass
    
    def _compute_mel_distance(self, ref_audio, test_audio):
        """Compute mel-spectrogram L2 distance"""
        # Implementation of mel-spectrogram comparison
        pass
```

## Expected Performance Outcomes

### Memory Reduction Targets

**Conservative Strategy (INT8 Primary):**
- T3 Model: 1.5GB → 750MB (50% reduction)
- S3Gen: 600MB → 300MB (50% reduction)
- Total VRAM: 3.85GB → 2.4GB (**38% reduction**)

**Balanced Strategy (Mixed Quantization):**
- T3 Model: 1.5GB → 400MB (73% reduction)
- S3Gen: 600MB → 300MB (50% reduction) 
- Total VRAM: 3.85GB → 1.9GB (**51% reduction**)

**Aggressive Strategy (NF4 + Double Quant):**
- T3 Model: 1.5GB → 300MB (80% reduction)
- S3Gen: 600MB → 150MB (75% reduction)
- Total VRAM: 3.85GB → 1.4GB (**64% reduction**)

### Performance Impact Projections

**Speed Impact:**
- NF4 4-bit: 0.9-1.1x baseline (potential slight slowdown)
- INT8 8-bit: 1.0-1.2x baseline (potential slight speedup)
- Mixed approach: ~1.0x baseline (negligible impact)

**Quality Preservation Targets:**
- Spectral distortion: <5% increase
- Perceptual quality: <2% MOS degradation
- Speaker similarity: >95% preservation
- Real-time factor: Maintain <0.8 RTF

## Risk Assessment & Mitigation

### High-Risk Components

**1. S3Gen Decoder Quantization**
- **Risk**: Audio quality degradation in vocoder
- **Mitigation**: Conservative 8-bit only, extensive A/B testing
- **Fallback**: Keep decoder in FP16, quantize flow layers only

**2. Training Compatibility**
- **Risk**: Existing checkpoints incompatible with quantized models
- **Mitigation**: Implement checkpoint conversion utilities
- **Strategy**: Support both quantized and non-quantized loading

**3. Dynamic Range Issues**
- **Risk**: Audio generation sensitive to outlier handling
- **Mitigation**: Calibration dataset for quantization statistics
- **Monitoring**: Real-time quality metric tracking

### Medium-Risk Areas

**1. Memory Fragmentation**
- **Risk**: Quantized models may increase fragmentation
- **Mitigation**: Optimized memory allocation patterns
- **Testing**: Long-running stability tests

**2. Hardware Compatibility**
- **Risk**: Quantization performance varies across GPUs
- **Mitigation**: Hardware-specific benchmarking
- **Support**: RTX 4090, RTX 3080/3090, A100 validation

### Low-Risk Areas

**1. Voice Encoder Quantization**
- **Risk**: Minimal - embedding extraction robust
- **Benefit**: High memory savings for low risk

**2. T3 Backbone Quantization**
- **Risk**: Low - transformers well-suited for quantization
- **Research**: Extensive literature on LLM quantization

## Implementation Timeline

### Phase 1: Infrastructure (Week 1-2)
- [x] Verify bitsandbytes installation and compatibility
- [ ] Implement quantization configuration classes
- [ ] Create component-specific quantization wrappers
- [ ] Basic integration with ChatterboxTTS loading

### Phase 2: Component Quantization (Week 2-3)
- [ ] T3 Llama backbone NF4 quantization
- [ ] S3Gen selective INT8 quantization  
- [ ] Voice encoder aggressive NF4 quantization
- [ ] End-to-end integration testing

### Phase 3: Optimization & Testing (Week 3-4)
- [ ] Double quantization implementation
- [ ] Dynamic quantization strategy
- [ ] Comprehensive benchmarking framework
- [ ] Quality assessment automation

### Phase 4: Production Ready (Week 4-5)
- [ ] Model conversion utilities
- [ ] Documentation and usage examples
- [ ] Performance comparison analysis
- [ ] Production deployment guidelines

## Quality Preservation Strategies

### Component-Specific Precision Control

```python
class PrecisionPreservationStrategy:
    """Strategies to maintain quality during quantization"""
    
    KEEP_FP16 = [
        # Critical for audio quality
        's3gen.hift.final_layer',
        's3gen.mel_extractor', 
        
        # Numerical stability critical
        'layernorm_layers',
        'softmax_operations',
        
        # Small impact, high quality benefit  
        'embedding_layers',
        'projection_heads'
    ]
    
    SAFE_FOR_NF4 = [
        # Well-suited for aggressive quantization
        'transformer.layers.*.self_attn',
        'transformer.layers.*.mlp',
        'voice_encoder.lstm',
        'voice_encoder.projection'
    ]
    
    SAFE_FOR_INT8 = [
        # Conservative quantization
        's3gen.flow.*',
        's3gen.decoder.conv_layers'
    ]
```

### Calibration Dataset Strategy

```python
def create_quantization_calibration_dataset():
    """Create calibration dataset for quantization statistics"""
    
    # Diverse audio samples for calibration
    calibration_samples = [
        # Different speakers
        'male_speaker_samples',
        'female_speaker_samples', 
        'child_speaker_samples',
        
        # Different content types
        'conversational_speech',
        'narrative_speech',
        'emotional_speech',
        
        # Different languages/accents
        'english_samples',
        'accented_english',
        'multilingual_samples'
    ]
    
    return prepare_calibration_data(calibration_samples)
```

## Integration with Existing Infrastructure

### Compatibility with Current Quantized Models

The bitsandbytes quantization approach complements existing infrastructure:

**Relationship to Current Quantization:**
- **Float16 weights**: Basic precision reduction → bitsandbytes provides deeper compression
- **Mixed precision**: Runtime optimization → bitsandbytes provides storage optimization
- **Combined approach**: Use bitsandbytes for storage + mixed precision for compute

**Migration Path:**
```python
def migrate_to_bitsandbytes():
    """Migration path from existing quantization"""
    
    # Step 1: Convert existing float16 models
    existing_model = ChatterboxTTS.from_local("quantized_models/float16_weights/", "cuda")
    
    # Step 2: Apply bitsandbytes quantization
    quantized_model = apply_bitsandbytes_quantization(existing_model, strategy='mixed')
    
    # Step 3: Validate quality preservation
    quality_metrics = validate_quantization_quality(existing_model, quantized_model)
    
    # Step 4: Save new quantized variant
    save_quantized_model(quantized_model, "quantized_models/bitsandbytes_nf4/")
```

### Performance Test Harness Integration

The existing `performance_test_harness.py` can be extended to include bitsandbytes variants:

```python
# Additional model configurations
BITSANDBYTES_CONFIGS = [
    ("NF4 4-bit", "quantized", "bitsandbytes_nf4"),
    ("INT8 8-bit", "quantized", "bitsandbytes_int8"), 
    ("Mixed Quant", "quantized", "bitsandbytes_mixed"),
    ("NF4 Double", "quantized", "bitsandbytes_nf4_double"),
]

def get_extended_model_configs():
    """Extended model configurations including bitsandbytes"""
    base_configs = get_model_configs()  # Existing configurations
    return base_configs + BITSANDBYTES_CONFIGS
```

## Monitoring & Validation Framework

### Automated Quality Gates

```python
class QuantizationQualityGates:
    """Automated quality gates for quantization validation"""
    
    QUALITY_THRESHOLDS = {
        'spectral_distortion_max': 0.05,    # 5% maximum degradation
        'mel_distance_max': 0.1,           # Mel-spectrogram similarity
        'perceptual_quality_min': 0.95,    # MOS preservation
        'speaker_similarity_min': 0.90,    # Voice consistency
        'rtf_max': 0.85,                   # Performance requirement
    }
    
    def validate_quantized_model(self, model, test_cases):
        """Run validation with automatic pass/fail"""
        results = self.run_quality_assessment(model, test_cases)
        
        passed_gates = []
        failed_gates = []
        
        for metric, threshold in self.QUALITY_THRESHOLDS.items():
            if self._check_threshold(results[metric], threshold, metric):
                passed_gates.append(metric)
            else:
                failed_gates.append(metric)
        
        return {
            'overall_pass': len(failed_gates) == 0,
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'detailed_results': results
        }
```

### Real-time Performance Monitoring

```python
class QuantizationPerformanceMonitor:
    """Real-time monitoring for quantized model performance"""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'vram_usage_gb': 6.0,          # Alert if VRAM > 6GB
            'generation_failures': 0.01,   # Alert if failure rate > 1%
            'rtf_degradation': 1.2,        # Alert if RTF > 1.2
        }
    
    def monitor_inference(self, model, input_text):
        """Monitor single inference with metrics collection"""
        start_time = time.time()
        vram_before = torch.cuda.memory_allocated() / (1024**3)
        
        try:
            output = model.generate(input_text)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            output = None
        
        vram_after = torch.cuda.memory_allocated() / (1024**3)
        generation_time = time.time() - start_time
        
        metrics = {
            'timestamp': start_time,
            'success': success,
            'error': error,
            'generation_time': generation_time,
            'vram_peak': vram_after,
            'vram_delta': vram_after - vram_before,
        }
        
        self.metrics_history.append(metrics)
        self._check_alerts(metrics)
        
        return output, metrics
```

## Rollback & Recovery Strategy

### Gradual Rollback Approach

```python
class QuantizationRollbackManager:
    """Manage rollback from quantized to full precision models"""
    
    def __init__(self, backup_model_path):
        self.backup_model_path = backup_model_path
        self.rollback_stages = [
            'disable_double_quantization',
            'rollback_voice_encoder',
            'rollback_s3gen',
            'rollback_t3_backbone',
            'full_precision_restore'
        ]
    
    def execute_rollback(self, stage=None):
        """Execute rollback to specified stage"""
        if stage is None:
            stage = 'full_precision_restore'
        
        rollback_methods = {
            'disable_double_quantization': self._disable_double_quant,
            'rollback_voice_encoder': self._rollback_ve,
            'rollback_s3gen': self._rollback_s3gen,
            'rollback_t3_backbone': self._rollback_t3,
            'full_precision_restore': self._restore_full_precision
        }
        
        return rollback_methods[stage]()
    
    def _restore_full_precision(self):
        """Emergency restore to full precision model"""
        return ChatterboxTTS.from_local(self.backup_model_path, device="cuda")
```

### Automatic Failure Detection

```python
def setup_automatic_rollback():
    """Setup automatic rollback on critical failures"""
    
    failure_conditions = [
        lambda metrics: metrics['generation_failures'] > 0.05,  # >5% failure rate
        lambda metrics: metrics['vram_usage'] > 20.0,           # >20GB VRAM (impossible)
        lambda metrics: metrics['quality_score'] < 0.8,        # <80% quality
        lambda metrics: metrics['rtf'] > 2.0,                  # >2.0 RTF
    ]
    
    def check_failure_conditions(metrics):
        for condition in failure_conditions:
            if condition(metrics):
                return True
        return False
    
    return check_failure_conditions
```

## Conclusion

Bitsandbytes quantization represents a transformative optimization opportunity for Chatterbox TTS, offering substantial memory reductions while maintaining audio quality. The combination of NF4 4-bit quantization for the Llama backbone and selective INT8 quantization for quality-sensitive components provides an optimal balance between efficiency and performance.

**Key Success Factors:**
1. **Component-specific strategy**: Different quantization approaches for different model components
2. **Quality-first approach**: Conservative quantization for audio-critical components  
3. **Comprehensive testing**: Automated quality gates and performance monitoring
4. **Gradual deployment**: Phased rollout with fallback capabilities

**Expected Impact:**
- **Memory reduction**: 50-70% VRAM savings on RTX 4090
- **Model size**: From 1.5GB to 380MB-760MB depending on strategy
- **Quality preservation**: <2% MOS degradation with proper calibration
- **Performance**: Neutral to slightly positive RTF impact

**Integration Benefits:**
- Builds upon existing quantized model infrastructure
- Compatible with current performance test harness
- Maintains API compatibility with existing code
- Provides path for future optimizations (QLoRA training, edge deployment)

The implementation should prioritize the **balanced mixed quantization strategy** for production deployment, providing substantial memory savings while maintaining the high audio quality standards expected from Chatterbox TTS. The aggressive NF4 strategy can be offered as an experimental option for resource-constrained environments.

This optimization aligns perfectly with the existing performance optimization roadmap and provides a solid foundation for deploying Chatterbox TTS on a wider range of hardware configurations while maintaining production-quality audio output.
# Vocos Vocoder Replacement Plan

## Executive Summary

This plan outlines the comprehensive replacement of HiFi-T-GAN vocoder with Vocos vocoder in the Chatterbox TTS system. Vocos offers significant performance advantages with **13x faster generation**, **~1GB VRAM reduction** (from 1.3GB to 220MB), and **single forward pass** generation compared to HiFi-T-GAN's 40+ autoregressive steps.

The replacement targets the mel-to-waveform conversion component within S3Gen while maintaining compatibility with the existing streaming architecture and quality standards.

---

## Current Vocoder Architecture Analysis

### Current Implementation: HiFi-T-GAN (HiFTGenerator)

**Location**: `src/chatterbox/models/s3gen/hifigan.py`, `src/chatterbox/models/s3gen/s3gen.py`

#### Architecture Overview
- **Base Class**: `HiFTGenerator` - Neural Source Filter + ISTFTNet architecture
- **Generation Method**: Autoregressive with 40+ steps per frame
- **Memory Usage**: ~1.3GB VRAM at fp16
- **Integration Point**: `S3Token2Wav.mel2wav` in `s3gen.py` line 230-237

#### Key Components
1. **Neural Source Filter**: F0-driven harmonic/noise source generation
2. **Upsampling Path**: ConvTranspose1d layers with ResBlocks
3. **ISTFT Integration**: Spectral coefficient generation + inverse STFT
4. **Streaming Support**: Cache-based source continuation for streaming

#### Current Performance Metrics (RTX 4090)
- **VRAM Usage**: 4.97GB peak (including full model)
- **Vocoder Component**: ~1.3GB dedicated
- **RTF**: ~0.77 (Real-Time Factor)
- **Generation Time**: ~8.35s average for creative content

#### Architecture Strengths
- Mature, well-tested implementation
- Good streaming support with source caching
- Integrated F0 prediction and synthesis
- Quality-proven in production

#### Architecture Limitations
- **High Memory Usage**: 1.3GB for vocoder alone
- **Autoregressive Bottleneck**: 40+ sequential steps per frame
- **Complexity**: Multi-component pipeline (source → upsample → ISTFT)

---

## Vocos Vocoder Analysis

### Architecture Overview
- **Type**: GAN-based Fourier-domain vocoder
- **Generation Method**: Single forward pass spectral coefficient generation
- **Memory Usage**: ~220MB at fp16 (83% reduction)
- **Speed**: 13x faster than HiFi-GAN, 70x faster than BigVGAN

### Technical Advantages

#### 1. **Fourier-Domain Processing**
- Generates spectral coefficients instead of time-domain samples
- Leverages Inverse Fast Fourier Transform (IFFT) for upsampling
- Eliminates need for transposed convolutions

#### 2. **Single Forward Pass**
- Non-autoregressive generation
- Parallel processing of entire sequence
- Massive speed improvement over iterative approaches

#### 3. **ConvNeXt Backbone**
- Modern architectural improvements
- Better efficiency than dilated convolutions
- Maintains low temporal resolution throughout network

#### 4. **Phase Handling Innovation**
- Unit circle activation function for phase estimation
- Natural phase wrapping without artifacts
- Improved phase coherence

### Performance Benchmarks
- **Speed**: 6,696x real-time on GPU, 169x on CPU
- **Quality**: UTMOS 3.734 (vs BigVGAN 3.749) - competitive quality
- **Memory**: 13.5M parameters (mel variant), 7.9M (EnCodec variant)

---

## Integration Requirements

### 1. **Dependency Management**

#### Installation Requirements
```bash
pip install vocos  # Core package
```

#### Version Compatibility
- PyTorch >= 1.9.0
- Python >= 3.7
- CUDA support (optional but recommended)

### 2. **Model Loading Integration**

#### Pre-trained Models Available
- `charactr/vocos-mel-24khz`: LibriTTS trained, 13.5M params
- `charactr/vocos-encodec-24khz`: DNS Challenge trained, 7.9M params

#### Target Integration Point
- **Primary**: `S3Token2Wav.mel2wav` replacement
- **File**: `src/chatterbox/models/s3gen/s3gen.py`
- **Lines**: 230-237, 260-261, 282-285

### 3. **Interface Compatibility**

#### Current Interface (HiFTGenerator)
```python
def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor) -> torch.Tensor:
    # Returns: (generated_speech, source_cache)
```

#### Vocos Interface
```python
def decode(self, mel: torch.Tensor) -> torch.Tensor:
    # Returns: generated_speech
```

#### Key Differences
- **Cache Source**: Vocos doesn't use autoregressive caching
- **Input Format**: Both accept mel spectrograms
- **Output**: Vocos single tensor vs HiFT tuple return

---

## Replacement Implementation Plan

### Phase 1: Core Integration (Week 1)

#### Step 1.1: Vocos Wrapper Implementation (2 days)
**Objective**: Create compatibility wrapper for Vocos

```python
class VocosWrapper(torch.nn.Module):
    """Compatibility wrapper for Vocos vocoder to match HiFT interface"""
    
    def __init__(self, model_name="charactr/vocos-mel-24khz"):
        super().__init__()
        from vocos import Vocos
        self.vocos = Vocos.from_pretrained(model_name)
        
    def inference(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = None):
        """Match HiFTGenerator interface"""
        # Ignore cache_source (not needed for non-autoregressive)
        audio = self.vocos.decode(speech_feat)
        # Return dummy cache for compatibility
        return audio, torch.zeros(1, 1, 0)
        
    def forward(self, batch: dict, device: torch.device):
        """Match training interface"""
        speech_feat = batch['speech_feat'].transpose(1, 2).to(device)
        audio = self.vocos.decode(speech_feat)
        # Return dummy f0 for compatibility
        return audio, None
```

#### Step 1.2: S3Gen Integration (2 days)
**File**: `src/chatterbox/models/s3gen/s3gen.py`

**Changes Required**:
1. **Import Addition** (line 30):
   ```python
   # Replace
   from .hifigan import HiFTGenerator
   # With
   from .vocos_wrapper import VocosWrapper
   ```

2. **Initialization Update** (lines 230-237):
   ```python
   # Replace HiFTGenerator initialization
   self.mel2wav = VocosWrapper(model_name="charactr/vocos-mel-24khz")
   ```

3. **Interface Compatibility**: No changes needed due to wrapper

#### Step 1.3: Device and Precision Handling (1 day)
```python
def __init__(self):
    super().__init__()
    self.mel2wav = VocosWrapper()
    
    # Ensure proper device placement
    if torch.cuda.is_available():
        self.mel2wav = self.mel2wav.to("cuda", dtype=torch.float16)
```

### Phase 2: Streaming Optimization (Week 2)

#### Step 2.1: Streaming Architecture Assessment (2 days)
**Current Streaming Mechanism**:
- HiFT uses `cache_source` for continuity between chunks
- Vocos single-pass generation eliminates need for caching
- Potential for improved streaming latency

**Analysis Required**:
- Impact on streaming chunk boundaries
- Audio continuity without autoregressive caching
- Latency improvements from single-pass generation

#### Step 2.2: Streaming Compatibility Testing (3 days)
**Test Cases**:
1. **Chunk Boundary Continuity**: Verify no artifacts at chunk transitions
2. **Latency Measurement**: Measure first-chunk latency improvement
3. **Real-time Performance**: Validate RTF improvements in streaming mode

### Phase 3: Performance Validation (Week 3)

#### Step 3.1: Benchmarking Infrastructure (2 days)
**Extend Performance Test Harness**:
```python
# Add to performance_test_harness.py
VOCOS_MODEL_PATH = Path("./models/vocos_integrated")

def load_vocos_model(device: str) -> ChatterboxTTS:
    """Load Chatterbox with Vocos vocoder"""
    return ChatterboxTTS.from_local(VOCOS_MODEL_PATH, device=device)
```

#### Step 3.2: Comprehensive Performance Testing (3 days)
**Metrics to Validate**:
- **VRAM Usage**: Target <3GB total (vs 4.97GB current)
- **Generation Speed**: Target <6s average (vs 8.35s current)
- **RTF**: Target <0.6 (vs 0.77 current)
- **Audio Quality**: PESQ/STOI comparison with original

### Phase 4: Quality Assurance (Week 4)

#### Step 4.1: Audio Quality Validation (3 days)
**Quality Tests**:
1. **Objective Metrics**: PESQ, STOI, MOS comparison
2. **Subjective Testing**: A/B testing with original HiFT audio
3. **Edge Cases**: Long audio, various voices, different languages

#### Step 4.2: Integration Testing (2 days)
**System-Level Tests**:
- Full pipeline integration (T3 → S3Gen → Vocos)
- Streaming functionality validation  
- Voice cloning compatibility
- Watermarking functionality preservation

---

## Compatibility Considerations

### 1. **Model Loading Compatibility**

#### Current Loading Pattern
```python
# checkpoints_grpo/merged_grpo_model/s3gen.pt contains HiFT weights
s3gen = S3Token2Wav()
s3gen.load_state_dict(torch.load("s3gen.pt"))
```

#### Vocos Integration Strategy
```python
# Option A: Hybrid checkpoint (recommended)
checkpoint = torch.load("s3gen.pt")
s3gen = S3Token2Wav()  # Now with Vocos
s3gen.load_state_dict(checkpoint, strict=False)  # Skip vocoder weights

# Option B: Full retraining with Vocos
# Retrain S3Gen with Vocos vocoder from scratch
```

### 2. **Training Pipeline Impact**

#### Current Training Dependencies
- `grpo.py` and `lora.py` expect HiFTGenerator interface
- F0 prediction integration built for HiFT architecture
- Loss functions may reference HiFT-specific components

#### Mitigation Strategy
- Wrapper maintains interface compatibility
- Training scripts require minimal modification
- F0 prediction can be disabled for Vocos (single-pass doesn't need it)

### 3. **Quantization Compatibility**

#### Current Quantized Models
- `quantized_models/mixed_precision/s3gen.pt` contains quantized HiFT
- Float16 optimizations applied to HiFT components

#### Vocos Quantization
- Vocos natively supports float16
- Smaller model size reduces quantization benefits
- May not need separate quantized variants

---

## Quality Validation Approach

### 1. **Objective Quality Metrics**

#### Audio Quality Assessment
```python
import pesq
import pystoi
import librosa

def evaluate_audio_quality(original_audio, vocos_audio, sample_rate):
    """Comprehensive audio quality evaluation"""
    
    # PESQ (Perceptual Evaluation of Speech Quality)
    pesq_score = pesq.pesq(sample_rate, original_audio, vocos_audio, 'wb')
    
    # STOI (Short-Time Objective Intelligibility)
    stoi_score = pystoi.stoi(original_audio, vocos_audio, sample_rate)
    
    # Spectral metrics
    original_spec = librosa.stft(original_audio)
    vocos_spec = librosa.stft(vocos_audio)
    
    spectral_similarity = np.corrcoef(
        np.abs(original_spec).flatten(),
        np.abs(vocos_spec).flatten()
    )[0, 1]
    
    return {
        'pesq': pesq_score,
        'stoi': stoi_score,
        'spectral_similarity': spectral_similarity
    }
```

#### Target Quality Thresholds
- **PESQ**: ≥ 3.5 (current baseline)
- **STOI**: ≥ 0.85 (current baseline)
- **Spectral Similarity**: ≥ 0.90

### 2. **Subjective Quality Testing**

#### A/B Testing Protocol
1. **Test Set**: 20 diverse text samples
2. **Voices**: 5 different reference voices
3. **Listeners**: 10+ human evaluators
4. **Metrics**: Naturalness, Similarity, Overall Quality
5. **Blind Testing**: Randomized presentation order

#### Quality Acceptance Criteria
- No statistically significant quality degradation
- Preference score ≥ 45% (neutral threshold)

---

## Performance Benchmarking

### 1. **Memory Usage Validation**

#### Target Metrics
```python
# Expected VRAM usage after Vocos replacement
EXPECTED_VRAM_REDUCTION = {
    'vocoder_component': 1300 - 220,  # 1080MB reduction
    'total_model': 4970 - 1080,      # ~3890MB total
    'percentage_reduction': '22% total VRAM reduction'
}
```

#### Validation Tests
- Peak VRAM during inference
- Steady-state memory usage
- Memory fragmentation analysis

### 2. **Speed Performance Validation**

#### Target Performance Improvements
```python
EXPECTED_SPEED_IMPROVEMENTS = {
    'generation_time': '8.35s → <6s (28% improvement)',
    'rtf': '0.77 → <0.6 (22% improvement)',
    'first_chunk_latency': 'Significant reduction (TBD)',
    'throughput': '13x theoretical maximum'
}
```

#### Benchmark Tests
- Single inference timing
- Batch processing performance
- Streaming latency measurements
- Long-form audio generation

---

## Rollback Strategy

### 1. **Configuration-Based Switching**

#### Implementation
```python
# src/chatterbox/models/s3gen/s3gen.py
class S3Token2Wav(S3Token2Mel):
    def __init__(self, use_vocos=True):
        super().__init__()
        
        if use_vocos:
            from .vocos_wrapper import VocosWrapper
            self.mel2wav = VocosWrapper()
        else:
            from .hifigan import HiFTGenerator
            f0_predictor = ConvRNNF0Predictor()
            self.mel2wav = HiFTGenerator(f0_predictor=f0_predictor)
```

#### Environment Variable Control
```bash
# Enable Vocos (default)
export CHATTERBOX_USE_VOCOS=true

# Fallback to HiFT
export CHATTERBOX_USE_VOCOS=false
```

### 2. **Model Checkpoint Compatibility**

#### Hybrid Checkpoint Strategy
- Maintain separate vocoder weights in checkpoints
- Allow selective loading based on configuration
- Preserve original HiFT weights for rollback

### 3. **Gradual Deployment**

#### Deployment Phases
1. **Development Environment**: Internal testing only
2. **Staging Environment**: Limited user testing
3. **A/B Testing**: Gradual user base expansion
4. **Full Deployment**: Complete migration

---

## Risk Assessment and Mitigation

### High-Risk Areas

#### 1. **Audio Quality Degradation**
- **Risk**: Vocos quality doesn't match HiFT standards
- **Mitigation**: Comprehensive quality testing, configurable fallback
- **Detection**: Automated quality monitoring in CI/CD

#### 2. **Streaming Compatibility Issues**
- **Risk**: Single-pass generation disrupts streaming continuity
- **Mitigation**: Thorough streaming validation, chunk boundary testing
- **Detection**: Real-time streaming tests in test harness

#### 3. **Training Pipeline Disruption**  
- **Risk**: Existing fine-tuning workflows break
- **Mitigation**: Interface compatibility wrapper, gradual migration
- **Detection**: Training script validation tests

### Medium-Risk Areas

#### 1. **Performance Regression**
- **Risk**: Unexpected performance issues in specific scenarios
- **Mitigation**: Comprehensive benchmarking, configuration switching
- **Detection**: Performance monitoring in test harness

#### 2. **Memory Usage Patterns**
- **Risk**: Different memory allocation patterns cause issues
- **Mitigation**: Memory profiling, fragmentation analysis
- **Detection**: VRAM monitoring in production

### Low-Risk Areas

#### 1. **Dependency Conflicts**
- **Risk**: Vocos dependencies conflict with existing packages
- **Mitigation**: Virtual environment testing, dependency pinning
- **Detection**: CI/CD dependency checks

---

## Implementation Timeline

### Week 1: Core Integration
- **Days 1-2**: Vocos wrapper implementation and testing
- **Days 3-4**: S3Gen integration and compatibility testing
- **Day 5**: Device/precision handling and initial validation

### Week 2: Streaming Optimization
- **Days 1-2**: Streaming architecture analysis and design
- **Days 3-5**: Streaming compatibility implementation and testing

### Week 3: Performance Validation  
- **Days 1-2**: Benchmarking infrastructure extension
- **Days 3-5**: Comprehensive performance testing and optimization

### Week 4: Quality Assurance
- **Days 1-3**: Audio quality validation and subjective testing
- **Days 4-5**: Integration testing and documentation

### Week 5: Deployment Preparation
- **Days 1-2**: Rollback strategy implementation
- **Days 3-4**: Production deployment preparation
- **Day 5**: Final validation and release preparation

---

## Success Metrics

### Performance Targets
- **VRAM Reduction**: ≥20% total model VRAM usage
- **Speed Improvement**: ≥25% generation time reduction  
- **RTF Improvement**: ≤0.6 Real-Time Factor
- **First Chunk Latency**: ≥30% reduction in streaming mode

### Quality Targets
- **PESQ Score**: ≥95% of original HiFT quality
- **Subjective Testing**: No significant quality degradation
- **Edge Case Handling**: 100% compatibility with existing test suite

### Integration Targets
- **Zero Downtime**: Seamless rollback capability
- **Training Compatibility**: Existing fine-tuning workflows preserved
- **API Compatibility**: No breaking changes to public interface

---

## Conclusion

The Vocos vocoder replacement represents a significant opportunity to improve Chatterbox TTS performance while reducing memory requirements. With careful implementation focusing on compatibility and quality validation, this upgrade can deliver:

- **13x speed improvement** in vocoder generation
- **~1GB VRAM reduction** (22% total system reduction)
- **Enhanced streaming performance** through single-pass generation
- **Maintained audio quality** through comprehensive testing

The phased implementation approach with robust rollback capabilities ensures low-risk deployment while maximizing the performance benefits of modern vocoder architecture.

The investment in this upgrade positions Chatterbox TTS for improved scalability and user experience, particularly beneficial for real-time streaming applications and resource-constrained deployment scenarios.
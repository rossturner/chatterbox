# Chatterbox TTS Optimization Guide

This document provides detailed information on optimization strategies for improving Chatterbox TTS performance, focusing on reducing generation time and voice cloning overhead.

## Pre-computed Voice Conditionals

### What Are Voice Conditionals?

Voice conditionals are the neural representations that tell Chatterbox TTS "how to sound like a specific speaker." When you provide a reference audio file, the system extracts several types of conditioning information:

1. **Speaker Embeddings** - High-level voice characteristics (timbre, accent, speaking style)
2. **Speech Tokens** - Semantic speech representations from the reference audio
3. **Acoustic Features** - Low-level audio characteristics for the S3Gen vocoder
4. **Emotion Control** - Exaggeration parameters for emotional expression

### The Conditioning Pipeline

```python
def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
    # 1. Load and resample reference audio
    s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
    ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
    
    # 2. Extract acoustic features for S3Gen vocoder
    s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
    
    # 3. Generate speech tokens for T3 model conditioning
    t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
    
    # 4. Extract speaker embedding using voice encoder
    ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
    
    # 5. Combine into conditioning object
    self.conds = Conditionals(t3_cond, s3gen_ref_dict)
```

### Why Pre-computation Matters

This conditioning pipeline is computationally expensive:
- **Audio I/O**: Loading and resampling WAV files
- **Neural inference**: Voice encoder forward pass
- **Speech tokenization**: S3Tokenizer processing
- **Feature extraction**: Acoustic analysis for vocoder

By pre-computing these once, you eliminate this overhead from every generation call.

### Using Pre-computed Conditionals

**Basic Usage:**
```python
# One-time setup: pre-compute Nicole's voice
model.prepare_conditionals("voices/nicole.wav", exaggeration=0.5)

# Fast generation: reuses cached conditionals
wav1 = model.generate("Hello, this is Nicole speaking.")
wav2 = model.generate("I can generate multiple sentences quickly.")
wav3 = model.generate("Because my voice is already cached!")
```

**Advanced: Multiple Voice Library**
```python
class VoiceLibrary:
    def __init__(self, model):
        self.model = model
        self.voices = {}
    
    def add_voice(self, name, audio_path, exaggeration=0.5):
        """Pre-compute and cache a voice"""
        self.model.prepare_conditionals(audio_path, exaggeration)
        self.voices[name] = {
            'conditionals': copy.deepcopy(self.model.conds),
            'exaggeration': exaggeration
        }
    
    def use_voice(self, name):
        """Switch to a pre-computed voice instantly"""
        if name in self.voices:
            self.model.conds = self.voices[name]['conditionals']
        else:
            raise ValueError(f"Voice '{name}' not found")

# Usage
library = VoiceLibrary(model)
library.add_voice("nicole", "test/nicole.wav", exaggeration=0.5)
library.add_voice("alice", "test/alice.wav", exaggeration=0.7)
library.add_voice("bob", "test/bob.wav", exaggeration=0.3)

# Instant voice switching
library.use_voice("nicole")
wav1 = model.generate("Speaking as Nicole")

library.use_voice("alice")
wav2 = model.generate("Now speaking as Alice")
```

**Persistent Caching:**
```python
# Save conditionals to disk
torch.save(model.conds, "nicole_voice_cache.pt")

# Load pre-computed conditionals (fast startup)
model.conds = torch.load("nicole_voice_cache.pt", map_location=device)
```

### Performance Impact

Pre-computing eliminates voice processing time entirely:
- **Before**: 17.69s generation (includes voice processing)
- **After**: ~6-8s generation (voice processing already done)
- **Speedup**: 2-3x faster for voice cloning

## Model Quantization

### What Is Quantization?

Quantization reduces model precision from 32-bit floats (fp32) to lower precision formats like 16-bit (fp16) or 8-bit integers (int8). This reduces memory usage and can significantly speed up inference, especially on modern hardware.

### Types of Quantization

**1. Dynamic Quantization (Easiest)**
```python
import torch.quantization

# Quantize linear layers to int8
model.t3 = torch.quantization.quantize_dynamic(
    model.t3, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

model.s3gen = torch.quantization.quantize_dynamic(
    model.s3gen,
    {torch.nn.Linear, torch.nn.Conv1d},
    dtype=torch.qint8
)
```

**2. Static Quantization (Better Performance)**
```python
# Requires calibration dataset
def calibrate_model(model, calibration_data):
    model.eval()
    with torch.no_grad():
        for data in calibration_data:
            model(data)

# Prepare model for quantization
model.t3.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model.t3 = torch.quantization.prepare(model.t3)

# Calibrate with representative data
calibrate_model(model.t3, calibration_dataset)

# Convert to quantized model
model.t3 = torch.quantization.convert(model.t3)
```

**3. Mixed Precision (GPU-optimized)**
```python
# Use Automatic Mixed Precision (AMP)
from torch.cuda.amp import autocast

with autocast():
    wav = model.generate(text)
```

### Expected Performance Gains

- **Memory**: 50-75% reduction
- **Speed**: 1.5-3x faster on CPU, 1.2-2x on GPU
- **Quality**: Minimal degradation with proper calibration

### Quantization Considerations

- **Quality vs Speed**: More aggressive quantization = faster but potentially lower quality
- **Hardware Support**: int8 works best on modern CPUs, fp16 on modern GPUs
- **Model Components**: Different parts may benefit from different quantization strategies

## Speaker-Specific Fine-tuning

### Concept

Instead of using complex conditioning at runtime, fine-tune the base model to "bake in" a specific speaker's characteristics. This creates a speaker-specialized model that generates that voice without needing reference audio.

### Fine-tuning Approaches

**1. LoRA (Low-Rank Adaptation)**
```python
from peft import LoraConfig, get_peft_model

# Add LoRA adapters to T3 model
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
)

model.t3 = get_peft_model(model.t3, lora_config)
```

**2. Full Fine-tuning**
```python
# Fine-tune entire T3 model for Nicole's voice
def fine_tune_for_speaker(model, speaker_data, epochs=10):
    optimizer = torch.optim.AdamW(model.t3.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch in speaker_data:
            text, target_tokens = batch
            
            # Forward pass with speaker's voice as target
            outputs = model.t3.inference(
                text_tokens=text,
                t3_cond=nicole_conditionals  # Fixed Nicole conditioning
            )
            
            loss = F.cross_entropy(outputs, target_tokens)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

**3. Embedding Space Adaptation**
```python
# Learn a speaker-specific embedding offset
class SpeakerAdapter(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.speaker_offset = nn.Parameter(torch.zeros(embed_dim))
    
    def forward(self, base_embedding):
        return base_embedding + self.speaker_offset

# Add to model
model.speaker_adapter = SpeakerAdapter(model.t3.embed_dim)

# During inference
speaker_embedding = model.speaker_adapter(base_embedding)
```

### Data Requirements

**Training Data Needed:**
- **Audio**: 30-60 minutes of clean speech from target speaker
- **Transcripts**: Accurate text transcriptions
- **Diversity**: Various emotions, speaking styles, content types

**Data Preparation:**
```python
def prepare_speaker_dataset(audio_files, transcripts):
    dataset = []
    for audio_file, transcript in zip(audio_files, transcripts):
        # Extract speech tokens
        speech_tokens = model.s3gen.tokenizer.forward([audio_file])
        
        # Tokenize text
        text_tokens = model.tokenizer.text_to_tokens(transcript)
        
        dataset.append((text_tokens, speech_tokens))
    
    return dataset
```

### Benefits

- **Speed**: No conditioning overhead - direct generation
- **Consistency**: More consistent voice reproduction
- **Quality**: Potentially higher quality than conditioning-based approach
- **Memory**: Smaller runtime memory footprint

### Drawbacks

- **Storage**: Need separate model for each speaker
- **Training**: Requires significant compute and data
- **Flexibility**: Less flexible than conditioning-based approach

## Hardware Optimizations

### GPU Memory Optimization

**1. Gradient Checkpointing**
```python
# Trade compute for memory
model.t3.gradient_checkpointing_enable()
model.s3gen.gradient_checkpointing_enable()
```

**2. Model Sharding**
```python
# Split large models across multiple GPUs
from torch.nn.parallel import DataParallel

if torch.cuda.device_count() > 1:
    model.t3 = DataParallel(model.t3)
    model.s3gen = DataParallel(model.s3gen)
```

**3. Memory-Efficient Attention**
```python
# Use flash attention for memory efficiency
from flash_attn import flash_attn_func

# Replace standard attention with flash attention
# (requires model architecture modifications)
```

### CPU Optimizations

**1. Thread Optimization**
```python
import os

# Optimize CPU threading
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)
```

**2. NUMA Awareness**
```python
# Pin to specific CPU cores for consistency
import psutil

def set_cpu_affinity():
    p = psutil.Process()
    # Use only performance cores
    p.cpu_affinity([0, 1, 2, 3])  # Adjust based on system
```

### Specialized Hardware

**1. Apple Silicon (MPS)**
```python
# Ensure MPS backend is properly utilized
if torch.backends.mps.is_available():
    device = "mps"
    # Use MPS-optimized operations
    torch.backends.mps.enable_fallback()
```

**2. NVIDIA Optimizations**
```python
# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True  # For consistent input sizes
torch.backends.cudnn.deterministic = False  # For maximum speed

# Use Tensor Cores when available
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**3. Intel Hardware**
```python
# Intel Extension for PyTorch
import intel_extension_for_pytorch as ipex

model.t3 = ipex.optimize(model.t3)
model.s3gen = ipex.optimize(model.s3gen)
```

### Memory Management

```python
def optimize_memory():
    # Clear cache between generations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Use memory mapping for large models
    torch.serialization.add_safe_globals({
        'collections.OrderedDict': collections.OrderedDict
    })
    
    # Enable memory profiling
    torch.cuda.memory._record_memory_history(
        enabled=True,
        alloc_trace_enabled=True,
        alloc_trace_record_context=True
    )
```

## Torch Compile Speedup

### What Is torch.compile?

`torch.compile` is PyTorch 2.0's JIT compiler that optimizes model execution by analyzing computation graphs and generating optimized kernels.

### Basic Usage

```python
# Compile individual models
model.t3 = torch.compile(model.t3, mode="reduce-overhead")
model.s3gen = torch.compile(model.s3gen, mode="reduce-overhead")
model.ve = torch.compile(model.ve, mode="reduce-overhead")
```

### Compilation Modes

```python
# Different optimization strategies
modes = {
    "default": torch.compile(model.t3),  # Balanced
    "reduce-overhead": torch.compile(model.t3, mode="reduce-overhead"),  # Minimize overhead
    "max-autotune": torch.compile(model.t3, mode="max-autotune"),  # Maximum optimization
    "max-autotune-no-cudagraphs": torch.compile(model.t3, mode="max-autotune-no-cudagraphs")
}
```

### Advanced Compilation Options

```python
# Custom compilation with specific backends
model.t3 = torch.compile(
    model.t3,
    backend="inductor",  # Default backend
    mode="max-autotune",
    fullgraph=True,  # Compile entire graph (more optimization)
    dynamic=False,   # Static shapes for maximum optimization
)

# For specific hardware
if torch.cuda.is_available():
    # CUDA-specific optimizations
    model.t3 = torch.compile(model.t3, backend="tensorrt")  # If TensorRT available
```

### Compilation Best Practices

**1. Warm-up Compilation**
```python
def warmup_compiled_model(model, example_input):
    """Trigger compilation with example input"""
    with torch.no_grad():
        # First call triggers compilation (slow)
        _ = model.generate("Warmup text")
        
        # Subsequent calls are fast
        for _ in range(3):
            _ = model.generate("Additional warmup")
```

**2. Input Shape Consistency**
```python
# Avoid dynamic shapes that trigger recompilation
def pad_to_fixed_length(text_tokens, max_length=512):
    if text_tokens.size(-1) < max_length:
        padding = max_length - text_tokens.size(-1)
        text_tokens = F.pad(text_tokens, (0, padding))
    return text_tokens[:, :max_length]
```

**3. Selective Compilation**
```python
# Only compile performance-critical parts
def selective_compile(model):
    # Compile heavy computational blocks
    model.t3.llama_model = torch.compile(model.t3.llama_model, mode="max-autotune")
    model.s3gen.flow = torch.compile(model.s3gen.flow, mode="reduce-overhead")
    
    # Leave lightweight components uncompiled
    # (preprocessing, postprocessing, etc.)
    return model
```

### Expected Performance Gains

- **CPU**: 1.2-2x speedup typical
- **GPU**: 1.3-3x speedup possible
- **Memory**: Usually slight reduction
- **Compilation Time**: 30s-5min first run (one-time cost)

### Troubleshooting Compilation

```python
import torch._dynamo as dynamo

# Debug compilation issues
dynamo.config.log_level = logging.DEBUG
dynamo.config.verbose = True

# Handle compilation failures gracefully
@torch.compile(backend="inductor", fullgraph=False)
def safe_generate(model, text):
    try:
        return model.generate(text)
    except torch._dynamo.exc.TorchRuntimeError:
        # Fallback to eager mode
        return model.generate.__wrapped__(text)
```

### Integration Example

```python
class OptimizedChatterboxTTS:
    def __init__(self, base_model):
        self.model = base_model
        self._compile_models()
        self._warmup()
    
    def _compile_models(self):
        """Compile performance-critical components"""
        self.model.t3 = torch.compile(
            self.model.t3, 
            mode="max-autotune",
            fullgraph=True
        )
        self.model.s3gen = torch.compile(
            self.model.s3gen,
            mode="reduce-overhead",
            dynamic=False
        )
    
    def _warmup(self):
        """Trigger compilation with dummy inputs"""
        print("Warming up compiled models...")
        with torch.no_grad():
            for _ in range(3):
                _ = self.model.generate("Compilation warmup text.")
        print("Warmup complete!")
    
    def generate(self, text, **kwargs):
        """Optimized generation with compiled models"""
        return self.model.generate(text, **kwargs)

# Usage
optimized_model = OptimizedChatterboxTTS(model)
# Now 1.5-3x faster generation
wav = optimized_model.generate("Fast compiled generation!")
```

## Performance Benchmarking

To measure the impact of these optimizations:

```python
import time
from contextlib import contextmanager

@contextmanager
def benchmark(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.2f}s")

# Benchmark different configurations
test_text = "This is a benchmark test sentence."

with benchmark("Baseline"):
    wav1 = model.generate(test_text)

with benchmark("Pre-computed conditionals"):
    model.prepare_conditionals("voice.wav")
    wav2 = model.generate(test_text)

with benchmark("Compiled + Pre-computed"):
    compiled_model = torch.compile(model, mode="reduce-overhead")
    wav3 = compiled_model.generate(test_text)
```

These optimizations can be combined for maximum performance improvement. Start with pre-computed conditionals and torch.compile for the biggest wins, then explore more advanced optimizations based on your specific requirements.
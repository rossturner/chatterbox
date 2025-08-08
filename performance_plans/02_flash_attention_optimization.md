# Flash-Attention2 + Triton-Fused MLP Optimization for Chatterbox TTS

## Executive Summary

This report provides a comprehensive implementation plan for optimizing the Chatterbox TTS system using Flash-Attention2 and Triton-fused MLP operations. Based on analysis of the current T3 model architecture, which uses a LlamaModel backbone with SDPA attention, we can achieve an estimated 25-40% per-token speedup on RTX 4090 through strategic attention and MLP optimizations.

## Current Architecture Analysis

### T3 Model Attention Implementation

The Chatterbox T3 model uses a **LlamaModel** from transformers with the following configuration:

**File**: `/home/ross/workspace/chatterbox-streaming/src/chatterbox/models/t3/llama_configs.py`
```python
LLAMA_520M_CONFIG_DICT = dict(
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    attn_implementation="sdpa",    # Current: Scaled Dot Product Attention
    head_dim=64,
    attention_dropout=0.0,
    # ... other config
)
```

**Current Attention Flow**:
1. **T3 Model** (`/home/ross/workspace/chatterbox-streaming/src/chatterbox/models/t3/t3.py`) creates LlamaModel backbone
2. **LlamaModel** uses SDPA (Scaled Dot Product Attention) implementation
3. **Input Processing**: Text + speech tokens are embedded and concatenated before transformer processing
4. **Sequence Lengths**: Handles variable text (max 512) + speech tokens (max 2048) = ~2560 total sequence length

### S3Gen Transformer Attention

The S3Gen model has its own attention implementation:

**File**: `/home/ross/workspace/chatterbox-streaming/src/chatterbox/models/s3gen/transformer/attention.py`
- **MultiHeadedAttention**: Standard implementation with manual QKV projection
- **RelPositionMultiHeadedAttention**: Relative positional encoding variant
- **Performance**: Uses standard PyTorch operations without kernel fusion

## Flash-Attention2 Integration Plan

### Phase 1: Environment Preparation

#### Prerequisites Verification
**Current Environment**:
- PyTorch 2.6.0 ✅ (Flash-Attention2 requires 2.2+)
- CUDA Support ✅ (RTX 4090 Ada architecture supported)
- Linux Environment ✅ (Primary support platform)

#### Installation Requirements
```bash
# Install Flash-Attention2 (requires compilation)
pip install flash-attn==2.6.3
# Alternative: pre-compiled wheel for faster installation
pip install flash-attn==2.8.2 --no-build-isolation
```

**Build Dependencies**:
- Ninja build system: `pip install ninja`
- CUDA toolkit compatible with PyTorch 2.6.0
- 8GB+ RAM for compilation (parallel builds can exceed memory)

### Phase 2: T3 Model Flash-Attention2 Integration

#### Implementation Strategy

**Target File**: `/home/ross/workspace/chatterbox-streaming/src/chatterbox/models/t3/llama_configs.py`

```python
# Enhanced configuration for Flash-Attention2
LLAMA_520M_FLASH_ATTN_CONFIG = dict(
    # Existing config...
    vocab_size=8,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=30,
    num_attention_heads=16,
    
    # Flash-Attention2 specific configuration
    attn_implementation="flash_attention_2",  # Changed from "sdpa"
    use_flash_attention_2=True,
    head_dim=64,  # Supported by Flash-Attention2
    attention_dropout=0.0,  # Flash-Attention2 supports dropout
    
    # Memory optimization
    torch_dtype="bfloat16",  # Optimal for RTX 4090 Ada architecture
    use_cache=True,
    
    # Existing config maintained
    max_position_embeddings=131072,
    tie_word_embeddings=False,
    hidden_act="silu",
    attention_bias=False,
    mlp_bias=False,
    # ... rest unchanged
)

LLAMA_CONFIGS = {
    "Llama_520M": LLAMA_520M_CONFIG_DICT,
    "Llama_520M_Flash": LLAMA_520M_FLASH_ATTN_CONFIG,  # New optimized config
}
```

#### Model Initialization Patch

**Target File**: `/home/ross/workspace/chatterbox-streaming/src/chatterbox/models/t3/t3.py`

```python
class T3(nn.Module):
    def __init__(self, hp=T3Config()):
        super().__init__()
        self.hp = hp
        
        # Select optimized config when Flash-Attention2 is available
        config_name = hp.llama_config_name
        if self._flash_attention_available():
            config_name = f"{hp.llama_config_name}_Flash"
            print("Flash-Attention2 detected: Using optimized configuration")
        
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[config_name])
        self.tfmr = LlamaModel(self.cfg)
        # ... rest unchanged
    
    def _flash_attention_available(self):
        """Check if Flash-Attention2 is available and compatible."""
        try:
            import flash_attn
            # Verify GPU compatibility (RTX 4090 = Compute Capability 8.9)
            if torch.cuda.is_available():
                gpu_capability = torch.cuda.get_device_capability()
                return gpu_capability[0] >= 8  # Ampere/Ada/Hopper support
            return False
        except ImportError:
            return False
```

### Phase 3: S3Gen Attention Optimization

#### Flash-Attention2 Integration for S3Gen

**Target File**: `/home/ross/workspace/chatterbox-streaming/src/chatterbox/models/s3gen/transformer/attention.py`

```python
class FlashMultiHeadedAttention(nn.Module):
    """Flash-Attention2 optimized Multi-Head Attention layer."""
    
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float, key_bias: bool = True):
        super().__init__()
        assert n_feat % n_head == 0
        
        self.d_k = n_feat // n_head
        self.h = n_head
        self.dropout_rate = dropout_rate
        
        # Combined QKV projection for better memory efficiency
        self.qkv_proj = nn.Linear(n_feat, n_feat * 3, bias=key_bias)
        self.out_proj = nn.Linear(n_feat, n_feat)
        
        # Flash-Attention2 import
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.flash_available = True
        except ImportError:
            print("Warning: Flash-Attention2 not available, falling back to standard attention")
            self.flash_available = False
            self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Flash-Attention2 optimized forward pass.
        
        Args:
            query, key, value: Input tensors (batch, seq_len, n_feat)
            mask: Attention mask (optional)
        
        Returns:
            torch.Tensor: Attention output (batch, seq_len, n_feat)
        """
        if not self.flash_available or not query.is_cuda:
            return self._standard_attention(query, key, value, mask)
        
        batch_size, seq_len, n_feat = query.shape
        
        # Combined QKV projection for memory efficiency
        if torch.equal(query, key) and torch.equal(key, value):  # Self-attention case
            qkv = self.qkv_proj(query)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q = self.qkv_proj(query)[:, :, :n_feat]
            k = self.qkv_proj(key)[:, :, n_feat:2*n_feat] 
            v = self.qkv_proj(value)[:, :, 2*n_feat:]
        
        # Reshape for Flash-Attention2: (batch, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.h, self.d_k)
        k = k.view(batch_size, seq_len, self.h, self.d_k)
        v = v.view(batch_size, seq_len, self.h, self.d_k)
        
        # Flash-Attention2 call
        out = self.flash_attn_func(
            q, k, v,
            dropout_p=self.dropout_rate if self.training else 0.0,
            softmax_scale=None,  # Use default 1/sqrt(head_dim)
            causal=False,  # S3Gen typically uses bidirectional attention
            return_attn_probs=False
        )
        
        # Reshape back and project
        out = out.view(batch_size, seq_len, n_feat)
        return self.out_proj(out)
    
    def _standard_attention(self, query, key, value, mask):
        """Fallback to standard attention when Flash-Attention2 unavailable."""
        # Existing MultiHeadedAttention logic
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)
```

## Triton-Fused MLP Optimization Plan

### Phase 4: Liger Kernel Integration

#### Installation and Setup

```bash
# Install Liger Kernel for Triton-fused operations
pip install liger-kernel
```

#### LlamaModel MLP Optimization

**Target**: Replace standard LlamaModel MLP layers with Liger-optimized versions

```python
# In T3 model initialization
from liger_kernel.transformers import AutoLigerKernelForCausalLM

class T3(nn.Module):
    def __init__(self, hp=T3Config()):
        super().__init__()
        self.hp = hp
        
        # Create config
        config_name = hp.llama_config_name
        if self._flash_attention_available():
            config_name = f"{hp.llama_config_name}_Flash"
        
        self.cfg = LlamaConfig(**LLAMA_CONFIGS[config_name])
        
        # Use Liger-optimized LlamaModel if available
        if self._liger_available():
            self.tfmr = AutoLigerKernelForCausalLM.from_config(self.cfg, apply_liger_kernel_to_llama=True)
            print("Liger Kernel detected: Using Triton-fused MLP operations")
        else:
            self.tfmr = LlamaModel(self.cfg)
        
        # ... rest unchanged
    
    def _liger_available(self):
        """Check if Liger Kernel is available."""
        try:
            import liger_kernel
            return True
        except ImportError:
            return False
```

#### Custom Triton-Fused Operations

For more granular control, implement custom Triton kernels:

```python
# New file: /home/ross/workspace/chatterbox-streaming/src/chatterbox/optimizations/triton_ops.py
import torch
import triton
import triton.language as tl

@triton.jit
def fused_silu_mul_kernel(
    x_ptr, gate_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for fused SiLU(x) * gate operation (SwiGLU component)."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    gate = tl.load(gate_ptr + offsets, mask=mask)
    
    # Fused SiLU * gate computation
    silu_x = x / (1.0 + tl.exp(-x))  # SiLU activation
    result = silu_x * gate
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

def fused_silu_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(x) * gate operation using Triton."""
    assert x.shape == gate.shape
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_silu_mul_kernel[grid](
        x, gate, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

class TritonOptimizedMLP(nn.Module):
    """Triton-optimized MLP layer for transformer models."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # Triton-fused SiLU * gate operation
        if x.is_cuda:
            intermediate = fused_silu_mul(up, gate)
        else:
            # CPU fallback
            intermediate = F.silu(up) * gate
        
        return self.down_proj(intermediate)
```

## Performance Benchmarking Strategy

### Comprehensive Testing Framework

```python
# New file: /home/ross/workspace/chatterbox-streaming/performance_flash_attention_test.py
import torch
import time
import numpy as np
from src.chatterbox.models.t3.t3 import T3
from src.chatterbox.models.t3.modules.t3_config import T3Config

class FlashAttentionBenchmark:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_configs = [
            {"seq_len": 512, "batch_size": 1},
            {"seq_len": 1024, "batch_size": 1},
            {"seq_len": 2048, "batch_size": 1},
            {"seq_len": 512, "batch_size": 4},
        ]
    
    def benchmark_attention_implementation(self, use_flash_attention=False):
        """Benchmark different attention implementations."""
        results = []
        
        for config in self.test_configs:
            seq_len = config["seq_len"]
            batch_size = config["batch_size"]
            
            # Configure model
            t3_config = T3Config()
            if use_flash_attention:
                t3_config.llama_config_name = "Llama_520M_Flash"
            
            model = T3(t3_config).to(self.device)
            model.eval()
            
            # Create dummy input
            text_tokens = torch.randint(0, 1000, (batch_size, seq_len // 4), device=self.device)
            speech_tokens = torch.randint(0, 1000, (batch_size, seq_len * 3 // 4), device=self.device)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model.forward(
                        t3_cond=model.cond_enc.empty_cond(),
                        text_tokens=text_tokens,
                        text_token_lens=torch.tensor([seq_len // 4] * batch_size),
                        speech_tokens=speech_tokens,
                        speech_token_lens=torch.tensor([seq_len * 3 // 4] * batch_size),
                        training=False
                    )
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            memory_usage = []
            
            for _ in range(10):
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()
                
                with torch.no_grad():
                    _ = model.forward(
                        t3_cond=model.cond_enc.empty_cond(),
                        text_tokens=text_tokens,
                        text_token_lens=torch.tensor([seq_len // 4] * batch_size),
                        speech_tokens=speech_tokens,
                        speech_token_lens=torch.tensor([seq_len * 3 // 4] * batch_size),
                        training=False
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
                memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)  # GB
            
            result = {
                "config": config,
                "implementation": "Flash-Attention2" if use_flash_attention else "SDPA",
                "avg_time": np.mean(times),
                "std_time": np.std(times),
                "avg_memory": np.mean(memory_usage),
                "throughput": seq_len * batch_size / np.mean(times),
            }
            results.append(result)
            
            del model
            torch.cuda.empty_cache()
        
        return results
    
    def run_full_benchmark(self):
        """Run complete benchmark comparing implementations."""
        print("Benchmarking SDPA (baseline)...")
        sdpa_results = self.benchmark_attention_implementation(use_flash_attention=False)
        
        print("Benchmarking Flash-Attention2...")
        flash_results = self.benchmark_attention_implementation(use_flash_attention=True)
        
        # Compare results
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        for sdpa, flash in zip(sdpa_results, flash_results):
            config = sdpa["config"]
            speedup = sdpa["avg_time"] / flash["avg_time"]
            memory_reduction = (sdpa["avg_memory"] - flash["avg_memory"]) / sdpa["avg_memory"] * 100
            
            print(f"\nSequence Length: {config['seq_len']}, Batch Size: {config['batch_size']}")
            print(f"  SDPA:          {sdpa['avg_time']:.4f}s ± {sdpa['std_time']:.4f}s, {sdpa['avg_memory']:.2f}GB")
            print(f"  Flash-Attn2:   {flash['avg_time']:.4f}s ± {flash['std_time']:.4f}s, {flash['avg_memory']:.2f}GB")
            print(f"  Speedup:       {speedup:.2f}x")
            print(f"  Memory:        {memory_reduction:+.1f}%")

if __name__ == "__main__":
    benchmark = FlashAttentionBenchmark()
    benchmark.run_full_benchmark()
```

## Expected Performance Improvements

### RTX 4090 Performance Projections

Based on research and similar implementations:

| Optimization | Expected Speedup | Memory Reduction | Implementation Effort |
|--------------|------------------|------------------|----------------------|
| Flash-Attention2 (T3 Model) | 25-40% | 10-20% | Medium |
| Flash-Attention2 (S3Gen) | 15-25% | 5-15% | High |
| Triton-Fused MLP | 10-20% | 5-10% | High |
| Combined Optimizations | 50-75% | 20-35% | High |

### Sequence Length Impact

- **Short Sequences (256-512)**: Minimal improvement due to overhead
- **Medium Sequences (1024-2048)**: 25-40% speedup expected
- **Long Sequences (4096+)**: Maximum benefit, up to 75% speedup

### Memory Scaling Benefits

- **Baseline SDPA**: O(n²) memory complexity
- **Flash-Attention2**: O(n) memory complexity
- **Critical for Streaming**: Enables longer context windows without OOM

## Implementation Risk Assessment

### Low Risk Elements

1. **Flash-Attention2 T3 Integration**: Drop-in replacement for SDPA
2. **Fallback Mechanisms**: Automatic fallback to standard attention when Flash-Attention2 unavailable
3. **Configuration-based**: No breaking changes to existing API

### Medium Risk Elements

1. **S3Gen Attention Refactoring**: Requires more significant code changes
2. **Memory Layout Changes**: Different tensor layouts between implementations
3. **Precision Sensitivity**: bfloat16 vs float16 compatibility

### High Risk Elements

1. **Custom Triton Kernels**: Kernel stability and correctness validation required
2. **Platform Dependency**: CUDA-specific optimizations limit CPU compatibility
3. **Library Version Conflicts**: Flash-Attention2 version compatibility with PyTorch 2.6.0

## Testing and Validation Strategy

### Phase 1: Functional Validation

```bash
# Test script to validate correctness
python -m pytest tests/test_flash_attention_correctness.py -v
```

Key validation points:
1. **Numerical Stability**: Compare outputs between SDPA and Flash-Attention2
2. **Gradient Correctness**: Validate backward pass equivalence
3. **Edge Cases**: Empty sequences, single tokens, maximum lengths

### Phase 2: Performance Validation

```bash
# Run performance benchmark
python performance_flash_attention_test.py
```

Expected timeline:
- **Week 1**: Flash-Attention2 installation and T3 integration
- **Week 2**: Performance benchmarking and optimization tuning
- **Week 3**: S3Gen attention optimization (optional)
- **Week 4**: Triton-fused MLP integration (advanced)

### Phase 3: Production Validation

1. **A/B Testing**: Run production workloads with both implementations
2. **Quality Assurance**: Audio quality comparison using existing test harness
3. **Stability Testing**: Extended runtime validation

## Implementation Timeline

### Immediate (Week 1)
- [ ] Install Flash-Attention2 and verify RTX 4090 compatibility
- [ ] Implement T3 model configuration for Flash-Attention2
- [ ] Basic functional testing

### Short-term (Weeks 2-3)
- [ ] Performance benchmarking framework
- [ ] S3Gen attention optimization
- [ ] Memory usage optimization

### Medium-term (Weeks 4-6)
- [ ] Triton-fused MLP integration
- [ ] Advanced kernel optimizations
- [ ] Production deployment validation

## Conclusion

Flash-Attention2 integration represents a high-impact optimization opportunity for the Chatterbox TTS system. The T3 model's use of LlamaModel with SDPA attention makes it an ideal candidate for Flash-Attention2 optimization, particularly given the RTX 4090's Ada architecture support.

The combination of Flash-Attention2 and Triton-fused MLP operations could deliver 50-75% performance improvements for typical TTS workloads, with the added benefit of reduced memory usage enabling longer context windows for streaming applications.

The implementation strategy prioritizes low-risk, high-impact optimizations first (T3 Flash-Attention2 integration) followed by more advanced optimizations (custom Triton kernels), ensuring a stable upgrade path with measurable performance gains at each stage.
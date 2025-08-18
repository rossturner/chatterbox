#!/usr/bin/env python3
"""
Optimized Performance Test Harness for Chatterbox TTS Models
with torch.compile optimizations applied upfront.

This version:
1. Applies torch.compile optimizations to each model during loading
2. Performs warmup compilation before testing
3. Measures true inference performance after compilation
4. Tracks compilation overhead separately
"""

import os
import json
import random
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torchaudio
import librosa
import numpy as np

# Set environment for better performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable TensorFloat32 for better performance
torch.set_float32_matmul_precision('high')

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3gen import S3GEN_SR


# Configuration constants
AUDIO_DATA_DIR = Path("./audio_data_v3")
OUTPUT_DIR = Path("./output")

# Model paths
BASE_MODEL_TYPE = "pretrained"
GRPO_V3_MODEL_PATH = Path("./models/nicole_v2/grpo_v3")
LORA_V2_2_MODEL_PATH = Path("./models/nicole_v2/lora_v2_2")

# Test parameters
NUM_TEST_CASES = 3
RANDOM_SEED = 42  # For reproducible results

# Optimization parameters
USE_TORCH_COMPILE = True
USE_BFLOAT16 = True
REDUCED_CACHE_LEN = 1200  # Reduced from 4096 for better performance
COMPILE_MODE = "reduce-overhead"  # or "max-autotune" for maximum optimization


@dataclass
class PerformanceResult:
    """Container for performance metrics from a single generation"""
    model_name: str
    reference_audio: str
    transcript_text: str
    generation_time: float
    audio_duration: float
    rtf: float
    tokens_per_second: float  # New metric
    vram_before_mb: float
    vram_peak_mb: float
    vram_after_mb: float
    vram_used_mb: float
    output_file: str


@dataclass
class TestCase:
    """Container for a single test case configuration"""
    reference_audio_path: Path
    transcript_text: str
    transcript_key: str


@dataclass
class OptimizedModelInfo:
    """Container for optimized model information"""
    model: ChatterboxTTS
    name: str
    load_time: float
    compilation_time: float  # Time spent on torch.compile warmup
    model_path: Optional[str] = None
    is_optimized: bool = False
    optimization_details: Dict = None


def setup_environment():
    """Initialize the testing environment"""
    print("=" * 80)
    print("Optimized Chatterbox TTS Performance Test Harness")
    print("with torch.compile optimizations")
    print("=" * 80)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BFloat16 support: {torch.cuda.is_bf16_supported()}")
        print(f"Float32 MatMul Precision: {torch.get_float32_matmul_precision()}")
    else:
        print("WARNING: Running on CPU - torch.compile optimizations will be limited")
    
    print(f"\nOptimization settings:")
    print(f"  torch.compile: {USE_TORCH_COMPILE}")
    print(f"  BFloat16: {USE_BFLOAT16 and torch.cuda.is_bf16_supported()}")
    print(f"  Reduced cache: {REDUCED_CACHE_LEN}")
    print(f"  Compile mode: {COMPILE_MODE}")
    
    return device


def clear_output_directory():
    """Clear and recreate the output directory"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory cleared: {OUTPUT_DIR}")


def load_available_audio_files() -> List[Path]:
    """Load all available .wav files from the audio data directory"""
    audio_files = list(AUDIO_DATA_DIR.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {AUDIO_DATA_DIR}")
    
    print(f"Found {len(audio_files)} audio files in {AUDIO_DATA_DIR}")
    return audio_files


def load_available_transcripts() -> Dict[str, str]:
    """Load all available transcript files"""
    txt_files = list(AUDIO_DATA_DIR.glob("*.txt"))
    transcripts = {}
    
    loaded_count = 0
    for txt_file in txt_files:
        try:
            wav_file = txt_file.with_suffix('.wav')
            if wav_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    transcript_text = f.read().strip()
                
                if transcript_text:
                    transcripts[wav_file.name] = transcript_text
                    loaded_count += 1
        except Exception as e:
            print(f"Warning: Failed to load transcript {txt_file}: {e}")
    
    print(f"Loaded {loaded_count} transcripts from individual .txt files")
    return transcripts


def generate_test_cases(num_cases: int) -> List[TestCase]:
    """Generate test cases with random audio/transcript pairs (mismatched)"""
    print(f"\nGenerating {num_cases} test cases...")
    
    audio_files = load_available_audio_files()
    transcripts = load_available_transcripts()
    
    audio_files_with_transcripts = [f for f in audio_files if f.name in transcripts]
    
    if len(audio_files_with_transcripts) < num_cases:
        raise ValueError(f"Not enough audio files with transcripts. Found {len(audio_files_with_transcripts)}, need {num_cases}")
    
    selected_audio = random.sample(audio_files_with_transcripts, num_cases)
    transcript_keys = list(transcripts.keys())
    selected_transcript_keys = random.sample(transcript_keys, num_cases)
    
    test_cases = []
    for i in range(num_cases):
        audio_path = selected_audio[i]
        transcript_key = selected_transcript_keys[i]
        
        if transcript_key == audio_path.name:
            transcript_key = selected_transcript_keys[(i + 1) % num_cases]
        
        transcript_text = transcripts[transcript_key]
        test_case = TestCase(
            reference_audio_path=audio_path,
            transcript_text=transcript_text,
            transcript_key=transcript_key
        )
        test_cases.append(test_case)
        
        print(f"  Test case {i+1}:")
        print(f"    Reference audio: {audio_path.name}")
        print(f"    Transcript from: {transcript_key}")
        print(f"    Text: {transcript_text[:60]}...")
    
    return test_cases


def apply_optimizations(model: ChatterboxTTS, device: str) -> Dict:
    """Apply all performance optimizations to a model"""
    optimization_details = {
        'bfloat16': False,
        'torch_compile': False,
        'reduced_cache': False,
        'compile_mode': None
    }
    
    if not hasattr(model, 't3') or model.t3 is None:
        print("    ⚠ Model doesn't have T3 component, skipping optimizations")
        return optimization_details
    
    # 1. Convert to BFloat16 if supported
    if USE_BFLOAT16 and device == "cuda":
        if torch.cuda.is_bf16_supported():
            print("    Applying BFloat16 optimization...")
            model.t3 = model.t3.to(dtype=torch.bfloat16)
            optimization_details['bfloat16'] = True
        else:
            print("    ⚠ BFloat16 not supported on this GPU")
    
    # 2. Patch inference to use reduced cache
    if REDUCED_CACHE_LEN:
        print(f"    Applying reduced cache optimization (max_cache_len={REDUCED_CACHE_LEN})...")
        original_inference = model.t3.inference
        def patched_inference(*args, **kwargs):
            kwargs['max_cache_len'] = REDUCED_CACHE_LEN
            return original_inference(*args, **kwargs)
        model.t3.inference = patched_inference
        optimization_details['reduced_cache'] = True
    
    # 3. Apply torch.compile if enabled
    if USE_TORCH_COMPILE and device == "cuda":
        if hasattr(model.t3, '_step_compilation_target'):
            print(f"    Applying torch.compile (mode={COMPILE_MODE})...")
            try:
                model.t3._step_compilation_target = torch.compile(
                    model.t3._step_compilation_target,
                    mode=COMPILE_MODE,
                    fullgraph=True
                )
                optimization_details['torch_compile'] = True
                optimization_details['compile_mode'] = COMPILE_MODE
            except Exception as e:
                print(f"    ⚠ torch.compile failed: {e}")
        else:
            print("    ⚠ Model doesn't support _step_compilation_target")
    
    return optimization_details


def warmup_model(model: ChatterboxTTS, test_case: TestCase, device: str) -> float:
    """Perform warmup runs to trigger torch.compile compilation"""
    print("    Warming up model (triggering compilation)...")
    
    # Use a short text for warmup
    warmup_text = "Hello, this is a warmup run."
    
    # Use the audio path directly with generate() - it handles conditionals internally
    audio_path = str(test_case.reference_audio_path)
    
    compilation_start = time.perf_counter()
    
    # First warmup - triggers compilation
    print("      Warmup 1 (initial compilation)...")
    start = time.perf_counter()
    try:
        _ = model.generate(warmup_text, audio_path, temperature=0.5, cfg_weight=0.5)
    except Exception as e:
        print(f"        Warning: Warmup 1 failed: {e}")
        return 0.0
    warmup1_time = time.perf_counter() - start
    print(f"        Time: {warmup1_time:.2f}s")
    
    # Second warmup - uses compiled code
    print("      Warmup 2 (using compiled code)...")
    start = time.perf_counter()
    try:
        _ = model.generate(warmup_text, audio_path, temperature=0.5, cfg_weight=0.5)
    except Exception as e:
        print(f"        Warning: Warmup 2 failed: {e}")
        return warmup1_time
    warmup2_time = time.perf_counter() - start
    print(f"        Time: {warmup2_time:.2f}s")
    
    compilation_time = time.perf_counter() - compilation_start
    
    # Estimate compilation overhead
    compilation_overhead = warmup1_time - warmup2_time
    print(f"      Estimated compilation overhead: {compilation_overhead:.2f}s")
    
    return compilation_time


def get_gpu_memory_mb() -> Optional[float]:
    """Get current GPU memory usage in MB"""
    if not torch.cuda.is_available():
        return None
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return torch.cuda.memory_allocated() / (1024 ** 2)


def force_cuda_cleanup():
    """Force CUDA cleanup to get accurate memory measurements"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()


def calculate_audio_duration(wav_tensor: torch.Tensor, sample_rate: int = S3GEN_SR) -> float:
    """Calculate audio duration in seconds from tensor"""
    if wav_tensor.dim() > 1:
        wav_tensor = wav_tensor.squeeze()
    return len(wav_tensor) / sample_rate


def estimate_tokens_per_second(generation_time: float, audio_duration: float) -> float:
    """Estimate tokens per second based on generation time and audio duration"""
    # Rough estimate: ~50 tokens per second of audio
    estimated_tokens = audio_duration * 50
    return estimated_tokens / generation_time if generation_time > 0 else 0


def get_model_configs() -> List[Tuple[str, str, Optional[str]]]:
    """Get list of model configurations to test"""
    return [
        ("Base Chatterbox", "pretrained", None),
        ("GRPO Fine-tuned (Nicole v2)", "local", str(GRPO_V3_MODEL_PATH)),
        ("LoRA Fine-tuned v2.2 (Nicole v2)", "local", str(LORA_V2_2_MODEL_PATH)),
    ]


def load_and_optimize_model(
    model_name: str, 
    model_type: str, 
    model_path: Optional[str], 
    device: str,
    test_case: TestCase  # Need a test case for warmup
) -> OptimizedModelInfo:
    """Load a model and apply optimizations including compilation warmup"""
    print(f"\n  Loading {model_name}...")
    
    # Load the base model
    start_time = time.time()
    if model_type == "pretrained":
        try:
            model = ChatterboxTTS.from_pretrained(device)
            load_time = time.time() - start_time
            print(f"    ✓ Loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"    ✗ Failed to load {model_name}: {e}")
            raise
    elif model_type == "local":
        if model_path is None:
            raise ValueError(f"Model path required for local model: {model_name}")
        
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            model = ChatterboxTTS.from_local(model_path_obj, device)
            load_time = time.time() - start_time
            print(f"    ✓ Loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"    ✗ Failed to load {model_name}: {e}")
            raise
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Apply optimizations
    print(f"  Optimizing {model_name}...")
    optimization_details = apply_optimizations(model, device)
    
    # Perform warmup if torch.compile was applied
    compilation_time = 0.0
    if optimization_details['torch_compile']:
        compilation_time = warmup_model(model, test_case, device)
    
    # Report optimization status
    print(f"  ✓ Optimization complete:")
    print(f"    - BFloat16: {optimization_details['bfloat16']}")
    print(f"    - torch.compile: {optimization_details['torch_compile']}")
    print(f"    - Reduced cache: {optimization_details['reduced_cache']}")
    if compilation_time > 0:
        print(f"    - Compilation time: {compilation_time:.2f}s")
    
    return OptimizedModelInfo(
        model=model,
        name=model_name,
        load_time=load_time,
        compilation_time=compilation_time,
        model_path=model_path if model_type == "local" else "from_pretrained",
        is_optimized=any(optimization_details.values()),
        optimization_details=optimization_details
    )


def run_single_optimized_test(
    model_info: OptimizedModelInfo,
    test_case: TestCase,
    test_id: str,
    device: str,
    vram_before_model_load: float = 0.0
) -> PerformanceResult:
    """Run a single performance test with an optimized model"""
    print(f"\n    Test {test_id}: {model_info.name}")
    print(f"      Reference: {test_case.reference_audio_path.name}")
    print(f"      Text: {test_case.transcript_text[:50]}...")
    
    # Clean up before measurement
    force_cuda_cleanup()
    
    # Memory before
    vram_before = get_gpu_memory_mb() or 0.0
    
    # Set seeds for reproducible generation
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Run generation with timing - pass audio path as second argument
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    wav_output = model_info.model.generate(
        test_case.transcript_text,
        str(test_case.reference_audio_path),  # Pass audio path as second argument
        temperature=0.5,
        cfg_weight=0.5
    )
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    generation_time = time.perf_counter() - start_time
    
    # Memory after
    force_cuda_cleanup()
    vram_after = get_gpu_memory_mb() or 0.0
    vram_peak = vram_after  # Approximation
    
    # Calculate metrics
    audio_duration = calculate_audio_duration(wav_output)
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    tokens_per_second = estimate_tokens_per_second(generation_time, audio_duration)
    vram_used = vram_peak - vram_before_model_load
    
    # Save output
    output_filename = f"{test_id}_{model_info.name.replace(' ', '_')}_{test_case.reference_audio_path.stem}.wav"
    output_path = OUTPUT_DIR / output_filename
    torchaudio.save(output_path, wav_output, S3GEN_SR)
    
    print(f"      ✓ Generated in {generation_time:.2f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
    print(f"      ✓ Tokens/s: {tokens_per_second:.1f} | VRAM: {vram_used:.1f}MB")
    print(f"      ✓ Saved: {output_filename}")
    
    return PerformanceResult(
        model_name=model_info.name,
        reference_audio=test_case.reference_audio_path.name,
        transcript_text=test_case.transcript_text,
        generation_time=generation_time,
        audio_duration=audio_duration,
        rtf=rtf,
        tokens_per_second=tokens_per_second,
        vram_before_mb=vram_before,
        vram_peak_mb=vram_peak,
        vram_after_mb=vram_after,
        vram_used_mb=vram_used,
        output_file=output_filename
    )


def unload_model(model_info: OptimizedModelInfo, device: str):
    """Unload a model and free VRAM"""
    print(f"    Unloading {model_info.name}...")
    
    # Thorough cleanup
    if hasattr(model_info.model, 't3') and model_info.model.t3 is not None:
        if hasattr(model_info.model.t3, 'cpu'):
            model_info.model.t3.cpu()
        del model_info.model.t3
        
    if hasattr(model_info.model, 's3gen') and model_info.model.s3gen is not None:
        if hasattr(model_info.model.s3gen, 'cpu'):
            model_info.model.s3gen.cpu()
        del model_info.model.s3gen
        
    if hasattr(model_info.model, 've') and model_info.model.ve is not None:
        if hasattr(model_info.model.ve, 'cpu'):
            model_info.model.ve.cpu()
        del model_info.model.ve
        
    if hasattr(model_info.model, 'cpu'):
        model_info.model.cpu()
        
    del model_info.model
    
    force_cuda_cleanup()
    time.sleep(0.5)
    force_cuda_cleanup()
    
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"    ✓ VRAM after unload: {memory_after:.1f}MB")


def run_optimized_performance_tests(test_cases: List[TestCase], device: str) -> Tuple[List[PerformanceResult], List[OptimizedModelInfo]]:
    """Run performance tests with optimized models"""
    model_configs = get_model_configs()
    
    print(f"\nRunning optimized performance tests...")
    print(f"  Models: {len(model_configs)}")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Total generations: {len(model_configs) * len(test_cases)}")
    print(f"  Approach: Load → Optimize → Compile → Test → Unload")
    
    results = []
    model_infos = []
    
    # Use first test case for warmup during optimization
    warmup_test_case = test_cases[0] if test_cases else None
    
    for model_idx, (model_name, model_type, model_path) in enumerate(model_configs):
        print(f"\n{'='*80}")
        print(f"TESTING MODEL {model_idx + 1}/{len(model_configs)}: {model_name}")
        print(f"{'='*80}")
        
        # Cleanup before loading
        print("  Performing pre-load cleanup...")
        force_cuda_cleanup()
        
        vram_before_load = get_gpu_memory_mb() or 0.0
        print(f"  VRAM before model load: {vram_before_load:.1f}MB")
        
        try:
            # Load and optimize model (includes compilation warmup)
            model_info = load_and_optimize_model(
                model_name, model_type, model_path, device, warmup_test_case
            )
            model_infos.append(model_info)
            
            vram_after_load = get_gpu_memory_mb() or 0.0
            model_vram_usage = vram_after_load - vram_before_load
            print(f"  VRAM after optimization: {vram_after_load:.1f}MB (+{model_vram_usage:.1f}MB)")
            
            # Run test cases
            model_results = []
            for case_idx, test_case in enumerate(test_cases):
                test_id = f"{model_idx + 1}.{case_idx + 1}"
                
                try:
                    result = run_single_optimized_test(
                        model_info, test_case, test_id, device, vram_before_load
                    )
                    results.append(result)
                    model_results.append(result)
                except Exception as e:
                    print(f"      ✗ Test {test_id} failed: {e}")
                    continue
                
                force_cuda_cleanup()
            
            # Model summary
            if model_results:
                avg_rtf = sum(r.rtf for r in model_results) / len(model_results)
                avg_tokens = sum(r.tokens_per_second for r in model_results) / len(model_results)
                avg_vram = sum(r.vram_used_mb for r in model_results) / len(model_results)
                print(f"\n  ✓ Model {model_name}: {len(model_results)} tests completed")
                print(f"    Average RTF: {avg_rtf:.3f}")
                print(f"    Average tokens/s: {avg_tokens:.1f}")
                print(f"    Average VRAM used: {avg_vram:.1f}MB")
                if model_info.compilation_time > 0:
                    print(f"    Compilation overhead: {model_info.compilation_time:.2f}s (one-time)")
            
            # Unload model
            unload_model(model_info, device)
            
        except Exception as e:
            print(f"  ✗ Failed to test model {model_name}: {e}")
            continue
    
    print(f"\n✓ Completed {len(results)} tests successfully")
    return results, model_infos


def print_optimized_results(results: List[PerformanceResult], model_infos: List[OptimizedModelInfo]):
    """Display comprehensive performance results with optimization details"""
    if not results:
        print("\nNo results to display")
        return
    
    print("\n" + "=" * 120)
    print("OPTIMIZED PERFORMANCE TEST RESULTS")
    print("=" * 120)
    
    # Group results by model
    models_results = {}
    for result in results:
        if result.model_name not in models_results:
            models_results[result.model_name] = []
        models_results[result.model_name].append(result)
    
    # Detailed results table
    print(f"\n{'Model':<35} {'Test':<4} {'Gen Time':<9} {'Audio':<7} {'RTF':<7} {'Tokens/s':<10} {'VRAM':<10}")
    print(f"{'Name':<35} {'ID':<4} {'(s)':<9} {'(s)':<7} {'':<7} {'':<10} {'(MB)':<10}")
    print("-" * 120)
    
    for model_name, model_results in models_results.items():
        for i, result in enumerate(model_results):
            test_id = f"{list(models_results.keys()).index(model_name) + 1}.{i + 1}"
            print(f"{model_name:<35} {test_id:<4} {result.generation_time:<9.2f} "
                  f"{result.audio_duration:<7.2f} {result.rtf:<7.3f} "
                  f"{result.tokens_per_second:<10.1f} {result.vram_used_mb:<10.1f}")
    
    # Summary statistics
    print("\n" + "-" * 120)
    print("SUMMARY STATISTICS")
    print("-" * 120)
    
    print(f"\n{'Model Name':<35} {'Tests':<6} {'Avg Gen':<9} {'Avg RTF':<8} {'Avg Tok/s':<11} {'Avg VRAM':<10}")
    print(f"{'':35} {'':6} {'Time (s)':<9} {'':8} {'':11} {'(MB)':<10}")
    print("-" * 120)
    
    for model_name, model_results in models_results.items():
        if not model_results:
            continue
            
        avg_gen_time = sum(r.generation_time for r in model_results) / len(model_results)
        avg_rtf = sum(r.rtf for r in model_results) / len(model_results)
        avg_tokens = sum(r.tokens_per_second for r in model_results) / len(model_results)
        avg_vram = sum(r.vram_used_mb for r in model_results) / len(model_results)
        
        print(f"{model_name:<35} {len(model_results):<6} {avg_gen_time:<9.2f} "
              f"{avg_rtf:<8.3f} {avg_tokens:<11.1f} {avg_vram:<10.1f}")
    
    print(f"\n✓ All tests completed successfully")
    print(f"✓ Output files saved to: {OUTPUT_DIR.absolute()}")


def main():
    """Main test harness execution"""
    try:
        # Setup environment
        device = setup_environment()
        clear_output_directory()
        
        print(f"\nTest Configuration:")
        print(f"  Random seed: {RANDOM_SEED}")
        print(f"  Test cases per model: {NUM_TEST_CASES}")
        
        # Generate test cases
        test_cases = generate_test_cases(NUM_TEST_CASES)
        
        # Run optimized performance tests
        results, model_infos = run_optimized_performance_tests(test_cases, device)
        
        # Display results
        print_optimized_results(results, model_infos)
        
        print(f"\n✓ Optimized performance testing completed!")
        print(f"✓ Generated {len(results)} audio files in {OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()
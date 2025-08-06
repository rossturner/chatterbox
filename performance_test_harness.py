#!/usr/bin/env python3
"""
Performance Test Harness for Chatterbox TTS Models

This script provides repeatable performance testing for inference generation
across different Chatterbox TTS model variants. Each run tests 3 models
against 3 random audio/transcript pairs (mismatched) for 9 total generations.

Measures:
- VRAM usage (before/during/after generation)  
- Generation time (excluding model loading and conditional preparation)
- Audio duration and Real-Time Factor (RTF)
"""

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

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.s3gen import S3GEN_SR


# Configuration constants
AUDIO_DATA_DIR = Path("./audio_data")
OUTPUT_DIR = Path("./output")
TRANSCRIPTS_CACHE = AUDIO_DATA_DIR / "transcripts_cache.json"

# Model paths
BASE_MODEL_TYPE = "pretrained"
GRPO_MODEL_PATH = Path("./checkpoints_grpo/merged_grpo_model")
QUANTIZED_MODEL_PATH = Path("./quantized_models/mixed_precision")

# Test parameters
NUM_TEST_CASES = 3
RANDOM_SEED = 42  # For reproducible results


@dataclass
class PerformanceResult:
    """Container for performance metrics from a single generation"""
    model_name: str
    reference_audio: str
    transcript_text: str
    generation_time: float
    audio_duration: float
    rtf: float
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


def setup_environment():
    """Initialize the testing environment"""
    print("=" * 60)
    print("Chatterbox TTS Performance Test Harness")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Running on CPU - VRAM measurements will not be available")
    
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


def load_transcripts_cache() -> Dict[str, Dict]:
    """Load the transcripts cache JSON file"""
    if not TRANSCRIPTS_CACHE.exists():
        raise FileNotFoundError(f"Transcripts cache not found: {TRANSCRIPTS_CACHE}")
    
    with open(TRANSCRIPTS_CACHE, 'r') as f:
        transcripts = json.load(f)
    
    print(f"Loaded {len(transcripts)} transcripts from cache")
    return transcripts


def generate_test_cases(num_cases: int) -> List[TestCase]:
    """Generate test cases with random audio/transcript pairs (mismatched)"""
    print(f"\nGenerating {num_cases} test cases...")
    
    # Load available data
    audio_files = load_available_audio_files()
    transcripts = load_transcripts_cache()
    
    # Randomly select audio files and transcript keys
    selected_audio = random.sample(audio_files, num_cases)
    transcript_keys = list(transcripts.keys())
    selected_transcript_keys = random.sample(transcript_keys, num_cases)
    
    # Create test cases, ensuring audio and transcripts are mismatched
    test_cases = []
    for i in range(num_cases):
        audio_path = selected_audio[i]
        transcript_key = selected_transcript_keys[i]
        
        # Ensure we don't accidentally match the audio file with its transcript
        audio_basename = audio_path.stem
        if transcript_key.startswith(audio_basename):
            # If matched, swap with next transcript (circular)
            transcript_key = selected_transcript_keys[(i + 1) % num_cases]
        
        transcript_data = transcripts[transcript_key]
        test_case = TestCase(
            reference_audio_path=audio_path,
            transcript_text=transcript_data["transcript"],
            transcript_key=transcript_key
        )
        test_cases.append(test_case)
        
        print(f"  Test case {i+1}:")
        print(f"    Reference audio: {audio_path.name}")
        print(f"    Transcript from: {transcript_key}")
        print(f"    Text: {transcript_data['transcript'][:60]}...")
    
    return test_cases


def get_gpu_memory_mb() -> Optional[float]:
    """Get current GPU memory usage in MB using nvidia-smi"""
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
        # Fallback to PyTorch method if nvidia-smi fails
        return torch.cuda.memory_allocated() / (1024 ** 2)


def get_peak_gpu_memory_mb() -> Optional[float]:
    """Get peak GPU memory usage in MB - use current for nvidia-smi"""
    # Since nvidia-smi doesn't track peak, we'll use current usage
    # This isn't perfect but gives us real VRAM usage
    return get_gpu_memory_mb()


def reset_peak_memory_stats():
    """Reset peak memory tracking - no-op for nvidia-smi approach"""
    # nvidia-smi doesn't have peak tracking, so we track current usage instead
    pass


def force_cuda_cleanup():
    """Force CUDA cleanup to get accurate memory measurements"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class VRAMSnapshot:
    """Container for VRAM measurements at a point in time"""
    current_mb: Optional[float]
    peak_mb: Optional[float]
    timestamp: float
    
    def __post_init__(self):
        if self.current_mb is None:
            self.current_mb = 0.0
        if self.peak_mb is None:
            self.peak_mb = 0.0


def take_vram_snapshot() -> VRAMSnapshot:
    """Take a snapshot of current VRAM usage"""
    return VRAMSnapshot(
        current_mb=get_gpu_memory_mb(),
        peak_mb=get_peak_gpu_memory_mb(),
        timestamp=time.time()
    )


@dataclass
class ModelInfo:
    """Container for loaded model information"""
    model: ChatterboxTTS
    name: str
    load_time: float
    model_path: Optional[str] = None


def load_base_model(device: str) -> ModelInfo:
    """Load the base Chatterbox TTS model from HuggingFace"""
    print(f"\n  Loading base model (from_pretrained)...")
    start_time = time.time()
    
    try:
        model = ChatterboxTTS.from_pretrained(device)
        load_time = time.time() - start_time
        
        print(f"    ✓ Loaded in {load_time:.2f}s")
        return ModelInfo(
            model=model,
            name="Base Chatterbox",
            load_time=load_time,
            model_path="from_pretrained"
        )
    except Exception as e:
        print(f"    ✗ Failed to load base model: {e}")
        raise


def load_grpo_model(device: str) -> ModelInfo:
    """Load the GRPO fine-tuned model"""
    print(f"\n  Loading GRPO model from {GRPO_MODEL_PATH}...")
    
    if not GRPO_MODEL_PATH.exists():
        raise FileNotFoundError(f"GRPO model not found: {GRPO_MODEL_PATH}")
    
    start_time = time.time()
    
    try:
        model = ChatterboxTTS.from_local(GRPO_MODEL_PATH, device)
        load_time = time.time() - start_time
        
        print(f"    ✓ Loaded in {load_time:.2f}s")
        return ModelInfo(
            model=model,
            name="GRPO Fine-tuned",
            load_time=load_time,
            model_path=str(GRPO_MODEL_PATH)
        )
    except Exception as e:
        print(f"    ✗ Failed to load GRPO model: {e}")
        raise


def load_quantized_model(device: str) -> ModelInfo:
    """Load the quantized model"""
    print(f"\n  Loading quantized model from {QUANTIZED_MODEL_PATH}...")
    
    if not QUANTIZED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Quantized model not found: {QUANTIZED_MODEL_PATH}")
    
    start_time = time.time()
    
    try:
        model = ChatterboxTTS.from_local(QUANTIZED_MODEL_PATH, device)
        load_time = time.time() - start_time
        
        print(f"    ✓ Loaded in {load_time:.2f}s")
        return ModelInfo(
            model=model,
            name="Mixed Precision Quantized",
            load_time=load_time,
            model_path=str(QUANTIZED_MODEL_PATH)
        )
    except Exception as e:
        print(f"    ✗ Failed to load quantized model: {e}")
        raise


def get_model_configs() -> List[Tuple[str, str, Optional[str]]]:
    """Get list of model configurations to test"""
    return [
        ("Base Chatterbox", "pretrained", None),
        ("GRPO Fine-tuned", "local", str(GRPO_MODEL_PATH)),
        ("Mixed Precision Quantized", "local", str(QUANTIZED_MODEL_PATH))
    ]


def load_single_model(model_name: str, model_type: str, model_path: Optional[str], device: str) -> ModelInfo:
    """Load a single model based on configuration"""
    print(f"\n  Loading {model_name}...")
    
    if model_type == "pretrained":
        return load_base_model(device)
    elif model_type == "local":
        if model_path is None:
            raise ValueError(f"Model path required for local model: {model_name}")
        
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        start_time = time.time()
        try:
            model = ChatterboxTTS.from_local(model_path_obj, device)
            load_time = time.time() - start_time
            
            print(f"    ✓ Loaded in {load_time:.2f}s")
            return ModelInfo(
                model=model,
                name=model_name,
                load_time=load_time,
                model_path=model_path
            )
        except Exception as e:
            print(f"    ✗ Failed to load {model_name}: {e}")
            raise
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def unload_model(model_info: ModelInfo, device: str):
    """Unload a model and free VRAM"""
    print(f"    Unloading {model_info.name}...")
    
    # Delete model references
    if hasattr(model_info.model, 't3'):
        del model_info.model.t3
    if hasattr(model_info.model, 's3gen'):
        del model_info.model.s3gen
    if hasattr(model_info.model, 've'):
        del model_info.model.ve
    del model_info.model
    
    # Force cleanup
    force_cuda_cleanup()
    
    # Report memory after cleanup
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"    ✓ VRAM after unload: {memory_after:.1f}MB")


def calculate_audio_duration(wav_tensor: torch.Tensor, sample_rate: int = S3GEN_SR) -> float:
    """Calculate audio duration in seconds from tensor"""
    if wav_tensor.dim() > 1:
        wav_tensor = wav_tensor.squeeze()
    return len(wav_tensor) / sample_rate


def run_single_test(
    model_info: ModelInfo, 
    test_case: TestCase, 
    test_id: str, 
    device: str
) -> PerformanceResult:
    """Run a single performance test case"""
    print(f"\n    Test {test_id}: {model_info.name}")
    print(f"      Reference: {test_case.reference_audio_path.name}")
    print(f"      Text: {test_case.transcript_text[:50]}...")
    
    # Prepare conditionals (not timed as part of generation)
    model_info.model.prepare_conditionals(str(test_case.reference_audio_path))
    
    # Clean up and prepare for accurate measurement
    force_cuda_cleanup()
    
    # Take before snapshot
    vram_before = take_vram_snapshot()
    
    # Run generation with timing and track max VRAM during generation
    start_time = time.time()
    max_vram_during_gen = vram_before.current_mb or 0.0
    
    # Start generation
    wav_output = model_info.model.generate(test_case.transcript_text)
    generation_time = time.time() - start_time
    
    # Take after snapshot  
    force_cuda_cleanup()
    vram_after = take_vram_snapshot()
    
    # For nvidia-smi, we can't track peak perfectly, so we'll check immediately after generation
    # The "peak" is approximated as the current usage right after generation
    vram_peak = vram_after.current_mb or 0.0
    
    # Calculate metrics
    audio_duration = calculate_audio_duration(wav_output)
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    vram_used = vram_peak - (vram_before.current_mb or 0.0)
    
    # Create output filename
    output_filename = f"{test_id}_{model_info.name.replace(' ', '_')}_{test_case.reference_audio_path.stem}.wav"
    output_path = OUTPUT_DIR / output_filename
    
    # Save wav file
    torchaudio.save(output_path, wav_output, S3GEN_SR)
    
    print(f"      ✓ Generated in {generation_time:.2f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
    print(f"      ✓ VRAM: {vram_before.current_mb:.1f}MB → {vram_after.current_mb:.1f}MB (peak: {vram_peak:.1f}MB)")
    print(f"      ✓ Saved: {output_filename}")
    
    return PerformanceResult(
        model_name=model_info.name,
        reference_audio=test_case.reference_audio_path.name,
        transcript_text=test_case.transcript_text,
        generation_time=generation_time,
        audio_duration=audio_duration,
        rtf=rtf,
        vram_before_mb=vram_before.current_mb or 0.0,
        vram_peak_mb=vram_peak,
        vram_after_mb=vram_after.current_mb or 0.0,
        vram_used_mb=vram_used,
        output_file=output_filename
    )


def run_performance_tests_sequential(test_cases: List[TestCase], device: str) -> List[PerformanceResult]:
    """Run performance tests one model at a time to properly track VRAM usage"""
    model_configs = get_model_configs()
    
    print(f"\nRunning sequential performance tests...")
    print(f"  Models: {len(model_configs)}")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Total generations: {len(model_configs) * len(test_cases)}")
    print(f"  Testing approach: Load → Test → Unload (one model at a time)")
    
    results = []
    
    for model_idx, (model_name, model_type, model_path) in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"TESTING MODEL {model_idx + 1}/{len(model_configs)}: {model_name}")
        print(f"{'='*60}")
        
        # Show VRAM before loading model
        vram_before_load = get_gpu_memory_mb() or 0.0
        print(f"  VRAM before model load: {vram_before_load:.1f}MB")
        
        try:
            # Load the model
            force_cuda_cleanup()
            reset_peak_memory_stats()
            
            model_info = load_single_model(model_name, model_type, model_path, device)
            
            # Show VRAM after loading model
            vram_after_load = get_gpu_memory_mb() or 0.0
            model_vram_usage = vram_after_load - vram_before_load
            print(f"  VRAM after model load: {vram_after_load:.1f}MB (+{model_vram_usage:.1f}MB)")
            
            # Run all test cases for this model
            model_results = []
            for case_idx, test_case in enumerate(test_cases):
                test_id = f"{model_idx + 1}.{case_idx + 1}"
                
                try:
                    result = run_single_test(model_info, test_case, test_id, device)
                    results.append(result)
                    model_results.append(result)
                except Exception as e:
                    print(f"      ✗ Test {test_id} failed: {e}")
                    continue
                    
                # Clean up between tests (but keep model loaded)
                force_cuda_cleanup()
            
            # Summary for this model
            if model_results:
                avg_rtf = sum(r.rtf for r in model_results) / len(model_results)
                avg_vram_used = sum(r.vram_used_mb for r in model_results) / len(model_results)
                print(f"  ✓ Model {model_name}: {len(model_results)} tests completed")
                print(f"    Average RTF: {avg_rtf:.3f}")
                print(f"    Average VRAM used per generation: {avg_vram_used:.1f}MB")
            
            # Unload the model to free VRAM
            unload_model(model_info, device)
            
        except Exception as e:
            print(f"  ✗ Failed to test model {model_name}: {e}")
            continue
    
    print(f"\n✓ Completed {len(results)} tests successfully")
    return results


def print_performance_results(results: List[PerformanceResult]):
    """Display comprehensive performance results in a formatted table"""
    if not results:
        print("\nNo results to display")
        return
    
    print("\n" + "=" * 100)
    print("PERFORMANCE TEST RESULTS")
    print("=" * 100)
    
    # Group results by model
    models_results = {}
    for result in results:
        if result.model_name not in models_results:
            models_results[result.model_name] = []
        models_results[result.model_name].append(result)
    
    # Print detailed results table
    print(f"\n{'Model':<25} {'Test':<4} {'Gen Time':<8} {'Audio':<6} {'RTF':<6} {'VRAM Used':<10} {'VRAM Peak':<10}")
    print(f"{'Name':<25} {'ID':<4} {'(s)':<8} {'(s)':<6} {'':<6} {'(MB)':<10} {'(MB)':<10}")
    print("-" * 100)
    
    for model_name, model_results in models_results.items():
        for i, result in enumerate(model_results):
            test_id = f"{list(models_results.keys()).index(model_name) + 1}.{i + 1}"
            print(f"{model_name:<25} {test_id:<4} {result.generation_time:<8.2f} "
                  f"{result.audio_duration:<6.2f} {result.rtf:<6.3f} "
                  f"{result.vram_used_mb:<10.1f} {result.vram_peak_mb:<10.1f}")
    
    # Calculate and print summary statistics
    print("\n" + "-" * 100)
    print("SUMMARY STATISTICS")
    print("-" * 100)
    
    print(f"\n{'Model Name':<25} {'Tests':<5} {'Avg Gen':<8} {'Avg RTF':<8} {'Avg VRAM':<10} {'Min RTF':<8} {'Max RTF':<8}")
    print(f"{'':25} {'':5} {'Time (s)':<8} {'':8} {'Used (MB)':<10} {'':8} {'':8}")
    print("-" * 100)
    
    for model_name, model_results in models_results.items():
        if not model_results:
            continue
            
        avg_gen_time = sum(r.generation_time for r in model_results) / len(model_results)
        avg_rtf = sum(r.rtf for r in model_results) / len(model_results)
        avg_vram = sum(r.vram_used_mb for r in model_results) / len(model_results)
        min_rtf = min(r.rtf for r in model_results)
        max_rtf = max(r.rtf for r in model_results)
        
        print(f"{model_name:<25} {len(model_results):<5} {avg_gen_time:<8.2f} "
              f"{avg_rtf:<8.3f} {avg_vram:<10.1f} {min_rtf:<8.3f} {max_rtf:<8.3f}")
    
    # Print model comparison
    if len(models_results) > 1:
        print(f"\n{'-' * 50}")
        print("MODEL COMPARISON")
        print(f"{'-' * 50}")
        
        baseline_results = list(models_results.values())[0]
        baseline_rtf = sum(r.rtf for r in baseline_results) / len(baseline_results)
        baseline_vram = sum(r.vram_used_mb for r in baseline_results) / len(baseline_results)
        
        print(f"\nUsing '{list(models_results.keys())[0]}' as baseline:")
        print(f"{'Model':<25} {'RTF vs Base':<12} {'VRAM vs Base':<12} {'Performance'}")
        print("-" * 70)
        
        for i, (model_name, model_results) in enumerate(models_results.items()):
            avg_rtf = sum(r.rtf for r in model_results) / len(model_results)
            avg_vram = sum(r.vram_used_mb for r in model_results) / len(model_results)
            
            if i == 0:
                rtf_diff = "baseline"
                vram_diff = "baseline"
                performance = "baseline"
            else:
                rtf_ratio = avg_rtf / baseline_rtf if baseline_rtf > 0 else float('inf')
                vram_ratio = avg_vram / baseline_vram if baseline_vram > 0 else float('inf')
                
                rtf_diff = f"{rtf_ratio:.2f}x" if rtf_ratio != float('inf') else "N/A"
                vram_diff = f"{vram_ratio:.2f}x" if vram_ratio != float('inf') else "N/A"
                
                if rtf_ratio < 1.0 and vram_ratio < 1.0:
                    performance = "BETTER"
                elif rtf_ratio < 1.0:
                    performance = "FASTER"
                elif vram_ratio < 1.0:
                    performance = "LIGHTER"
                else:
                    performance = "SLOWER/HEAVIER"
            
            print(f"{model_name:<25} {rtf_diff:<12} {vram_diff:<12} {performance}")
    
    # Print output files summary
    print(f"\n{'-' * 50}")
    print("OUTPUT FILES")
    print(f"{'-' * 50}")
    print(f"Total files generated: {len(results)}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    for result in results:
        print(f"  {result.output_file}")


def main():
    """Main test harness execution"""
    try:
        # Setup environment
        device = setup_environment()
        clear_output_directory()
        
        print(f"\nTest Configuration:")
        print(f"  Random seed: {RANDOM_SEED}")
        print(f"  Test cases per model: {NUM_TEST_CASES}")
        print(f"  Total generations: {NUM_TEST_CASES * 3}")
        
        # Generate test cases
        test_cases = generate_test_cases(NUM_TEST_CASES)
        
        # Run performance tests sequentially (load → test → unload)
        results = run_performance_tests_sequential(test_cases, device)
        
        # Display results
        print_performance_results(results)
        
        print(f"\n✓ Performance testing completed successfully!")
        print(f"✓ Generated {len(results)} audio files in {OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()


# Example usage:
# python3 performance_test_harness.py
# 
# The script will:
# 1. Randomly select 3 audio files and 3 transcripts (mismatched)  
# 2. For each model: Load → Test 3 cases → Unload (sequential approach)
# 3. Models tested: Base, GRPO fine-tuned, Mixed precision quantized
# 4. Generate 9 total audio files (3 models x 3 test cases)
# 5. Measure VRAM usage (per model load + per generation), timing, and RTF
# 6. Display comprehensive performance comparison tables
# 7. Save all generated audio files to ./output/ directory
#!/usr/bin/env python3
"""
Simple test to demonstrate torch.compile performance with the optimizations.
Also saves generated audio files to ./output_optimized directory.
"""

import os
import time
import torch
import torchaudio
from pathlib import Path

# Set environment for better performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')

from src.chatterbox.tts import ChatterboxTTS
from src.chatterbox.models.s3gen import S3GEN_SR


def test_optimized_performance():
    print("=" * 80)
    print("TORCH.COMPILE OPTIMIZED PERFORMANCE TEST")
    print("=" * 80)
    
    # Configuration
    model_path = "./models/nicole_v1/base_grpo"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_text = "Hello world, this is a test."
    output_dir = Path("./output_optimized")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    # Find reference audio
    audio_files = list(Path("audio_data_v2").glob("*.wav"))
    if not audio_files:
        print("ERROR: No audio files found")
        return
    reference_audio = str(audio_files[0])
    
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Reference: {Path(reference_audio).name}")
    print()
    
    # Load model
    print("Loading model...")
    model = ChatterboxTTS.from_local(model_path, device)
    
    # Apply optimizations
    print("\nApplying optimizations...")
    
    # 1. BFloat16
    if torch.cuda.is_bf16_supported():
        print("  - Converting to BFloat16")
        model.t3 = model.t3.to(dtype=torch.bfloat16)
    
    # 2. Reduced cache
    print("  - Setting reduced cache (600)")
    original_inference = model.t3.inference
    def patched_inference(*args, **kwargs):
        kwargs['max_cache_len'] = 600
        return original_inference(*args, **kwargs)
    model.t3.inference = patched_inference
    
    # 3. Apply torch.compile
    if hasattr(model.t3, '_step_compilation_target'):
        print("  - Applying torch.compile")
        model.t3._step_compilation_target = torch.compile(
            model.t3._step_compilation_target,
            mode="reduce-overhead",
            fullgraph=True
        )
    
    print("\n✓ Optimizations applied")
    
    # Warmup
    print("\n" + "-" * 80)
    print("WARMUP (Triggering Compilation)")
    print("-" * 80)
    
    print("Warmup 1 (compilation)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    warmup1_wav = model.generate(test_text, reference_audio, temperature=0.5, cfg_weight=0.5)
    torch.cuda.synchronize()
    warmup1_time = time.perf_counter() - start
    print(f"  Time: {warmup1_time:.2f}s")
    
    # Save warmup 1 audio
    warmup1_file = output_dir / "warmup1_compiled.wav"
    torchaudio.save(warmup1_file, warmup1_wav, S3GEN_SR)
    print(f"  Saved: {warmup1_file.name}")
    
    print("\nWarmup 2 (compiled)...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    warmup2_wav = model.generate(test_text, reference_audio, temperature=0.5, cfg_weight=0.5)
    torch.cuda.synchronize()
    warmup2_time = time.perf_counter() - start
    print(f"  Time: {warmup2_time:.2f}s")
    
    # Save warmup 2 audio
    warmup2_file = output_dir / "warmup2_compiled.wav"
    torchaudio.save(warmup2_file, warmup2_wav, S3GEN_SR)
    print(f"  Saved: {warmup2_file.name}")
    
    compilation_overhead = warmup1_time - warmup2_time
    print(f"\nCompilation overhead: {compilation_overhead:.2f}s")
    
    # Performance test
    print("\n" + "-" * 80)
    print("PERFORMANCE TEST (3 runs)")
    print("-" * 80)
    
    total_time = 0
    generated_wavs = []
    
    for i in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        wav = model.generate(test_text, reference_audio, temperature=0.5, cfg_weight=0.5)
        torch.cuda.synchronize()
        gen_time = time.perf_counter() - start
        
        # Save the generated audio (outside of timing)
        output_file = output_dir / f"test_run{i+1}_optimized.wav"
        torchaudio.save(output_file, wav, S3GEN_SR)
        generated_wavs.append((output_file, wav))
        
        # Estimate tokens
        audio_duration = wav.shape[-1] / 24000
        estimated_tokens = int(audio_duration * 50)
        tokens_per_sec = estimated_tokens / gen_time
        
        print(f"Run {i+1}: {gen_time:.2f}s, ~{estimated_tokens} tokens, {tokens_per_sec:.1f} tokens/s")
        print(f"  Saved: {output_file.name}")
        total_time += gen_time
    
    avg_time = total_time / 3
    print(f"\nAverage generation time: {avg_time:.2f}s")
    
    # Calculate overall tokens/s using the last generated wav
    audio_duration = generated_wavs[-1][1].shape[-1] / 24000
    estimated_tokens = int(audio_duration * 50)
    avg_tokens_per_sec = estimated_tokens / avg_time
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average generation time: {avg_time:.2f}s")
    print(f"Average tokens/second: {avg_tokens_per_sec:.1f}")
    print(f"Compilation overhead: {compilation_overhead:.2f}s (one-time)")
    
    if avg_tokens_per_sec >= 100:
        print(f"\n✓ SUCCESS: Achieved {avg_tokens_per_sec:.1f} tokens/s!")
    else:
        print(f"\n⚠ Performance: {avg_tokens_per_sec:.1f} tokens/s")
    
    # Summary of saved files
    print("\n" + "-" * 80)
    print("SAVED AUDIO FILES")
    print("-" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  - {warmup1_file.name} (warmup 1, with compilation)")
    print(f"  - {warmup2_file.name} (warmup 2, compiled)")
    for i, (file_path, _) in enumerate(generated_wavs, 1):
        print(f"  - {file_path.name} (test run {i})")
    print(f"\nTotal files saved: {2 + len(generated_wavs)}")


if __name__ == "__main__":
    test_optimized_performance()
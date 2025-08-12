#!/usr/bin/env python3
"""
Optimized model loader for Chatterbox TTS Server.
Applies torch.compile optimizations for 3-4x performance improvement.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
import glob

import torch

# Set environment for better performance
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')

from ..chatterbox.tts import ChatterboxTTS

logger = logging.getLogger(__name__)


class OptimizedModelLoader:
    """
    Handles loading and optimizing Chatterbox TTS models with:
    - BFloat16 precision
    - Reduced cache size for better performance
    - torch.compile optimizations
    - One-time warmup compilation
    """
    
    # Optimization settings
    USE_BFLOAT16 = True
    REDUCED_CACHE_LEN = 1200  # Reduced from 4096 for better performance
    COMPILE_MODE = "max-autotune"  # or "reduce-overhead" for faster compilation
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the optimized model loader.
        
        Args:
            device: Device to load model on (cuda/cpu)
        """
        self.device = device
        self.is_cuda = device == "cuda" and torch.cuda.is_available()
        self.supports_bfloat16 = self.is_cuda and torch.cuda.is_bf16_supported()
        
        # Log optimization capabilities
        logger.info(f"OptimizedModelLoader initialized on {device}")
        if self.is_cuda:
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  BFloat16 support: {self.supports_bfloat16}")
            logger.info(f"  Float32 MatMul Precision: {torch.get_float32_matmul_precision()}")
        else:
            logger.info("  Running on CPU - optimizations will be limited")
    
    def load_and_optimize(
        self, 
        model_type: str, 
        model_path: Optional[Path] = None
    ) -> Tuple[ChatterboxTTS, dict]:
        """
        Load a model and apply all optimizations.
        
        Args:
            model_type: Type of model (base/grpo/quantized)
            model_path: Path to local model (for grpo/quantized)
            
        Returns:
            Tuple of (optimized_model, optimization_info)
        """
        logger.info(f"Loading {model_type} model...")
        
        # Load the base model
        start_time = time.time()
        model = self._load_base_model(model_type, model_path)
        load_time = time.time() - start_time
        logger.info(f"  Model loaded in {load_time:.2f}s")
        
        # Apply optimizations
        optimization_info = self._apply_optimizations(model)
        
        # Report optimization status
        logger.info("Optimization complete:")
        logger.info(f"  BFloat16: {optimization_info['bfloat16']}")
        logger.info(f"  torch.compile: {optimization_info['torch_compile']}")
        logger.info(f"  Reduced cache: {optimization_info['reduced_cache']} ({self.REDUCED_CACHE_LEN} tokens)")
        if optimization_info['torch_compile']:
            logger.info(f"  Compile mode: {optimization_info['compile_mode']}")
        
        optimization_info['load_time'] = load_time
        
        return model, optimization_info
    
    def _load_base_model(
        self, 
        model_type: str, 
        model_path: Optional[Path] = None
    ) -> ChatterboxTTS:
        """
        Load the base model without optimizations.
        
        Args:
            model_type: Type of model
            model_path: Path to local model
            
        Returns:
            Loaded ChatterboxTTS model
        """
        if model_type == "base":
            # Load from HuggingFace
            model = ChatterboxTTS.from_pretrained(self.device)
            logger.info("  Loaded base model from HuggingFace")
            
        elif model_type in ["grpo", "quantized"]:
            # Load from local path
            if not model_path:
                raise ValueError(f"Model path required for {model_type} model")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            model = ChatterboxTTS.from_local(model_path, self.device)
            logger.info(f"  Loaded {model_type} model from {model_path}")
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    def _apply_optimizations(self, model: ChatterboxTTS) -> dict:
        """
        Apply all performance optimizations to the model.
        
        Args:
            model: ChatterboxTTS model to optimize
            
        Returns:
            Dictionary with optimization details
        """
        optimization_info = {
            'bfloat16': False,
            'torch_compile': False,
            'reduced_cache': False,
            'compile_mode': None
        }
        
        if not hasattr(model, 't3') or model.t3 is None:
            logger.warning("Model doesn't have T3 component, skipping optimizations")
            return optimization_info
        
        # 1. Convert to BFloat16 if supported
        if self.USE_BFLOAT16 and self.supports_bfloat16:
            logger.info("  Applying BFloat16 optimization...")
            model.t3 = model.t3.to(dtype=torch.bfloat16)
            optimization_info['bfloat16'] = True
        elif self.USE_BFLOAT16:
            logger.info("  BFloat16 not supported on this device, using default precision")
        
        # 2. Patch inference to use reduced cache
        if self.REDUCED_CACHE_LEN:
            logger.info(f"  Applying reduced cache optimization (max_cache_len={self.REDUCED_CACHE_LEN})...")
            original_inference = model.t3.inference
            
            def patched_inference(*args, **kwargs):
                kwargs['max_cache_len'] = self.REDUCED_CACHE_LEN
                return original_inference(*args, **kwargs)
            
            model.t3.inference = patched_inference
            optimization_info['reduced_cache'] = True
        
        # 3. Apply torch.compile if enabled and on CUDA
        if self.is_cuda:
            if hasattr(model.t3, '_step_compilation_target'):
                logger.info(f"  Applying torch.compile (mode={self.COMPILE_MODE})...")
                try:
                    model.t3._step_compilation_target = torch.compile(
                        model.t3._step_compilation_target,
                        mode=self.COMPILE_MODE,
                        fullgraph=True
                    )
                    optimization_info['torch_compile'] = True
                    optimization_info['compile_mode'] = self.COMPILE_MODE
                except Exception as e:
                    logger.warning(f"  torch.compile failed: {e}")
            else:
                logger.info("  Model doesn't support _step_compilation_target, skipping torch.compile")
        else:
            logger.info("  Skipping torch.compile (CPU mode)")
        
        return optimization_info
    
    def warmup_model(
        self, 
        model: ChatterboxTTS, 
        warmup_text: str = "Hello, this is a warmup run to trigger compilation.",
        warmup_audio_path: Optional[str] = None
    ) -> float:
        """
        Perform warmup runs to trigger torch.compile compilation.
        
        Args:
            model: Model to warm up
            warmup_text: Text to use for warmup
            warmup_audio_path: Optional audio path for warmup (if None, will find one)
            
        Returns:
            Total warmup/compilation time in seconds
        """
        logger.info("Warming up model (triggering compilation)...")
        
        # Find an audio file for warmup if not provided
        if warmup_audio_path is None:
            # Try to find any available audio file
            search_paths = [
                "audio_data/*.wav",
                "audio_data_v2/*.wav",
                "configs/voice_samples/**/*.wav",
                "configs/voice_samples/*.wav"
            ]
            
            audio_files = []
            for pattern in search_paths:
                audio_files.extend(glob.glob(pattern, recursive=True))
            
            if audio_files:
                warmup_audio_path = audio_files[0]
                logger.info(f"  Using audio file for warmup: {warmup_audio_path}")
            else:
                logger.warning("  No audio files found for warmup, model may not compile properly")
                return 0.0
        
        compilation_start = time.perf_counter()
        
        # First warmup - triggers compilation
        logger.info("  Warmup 1 (initial compilation)...")
        start = time.perf_counter()
        try:
            if warmup_audio_path:
                _ = model.generate(warmup_text, warmup_audio_path, temperature=0.5, cfg_weight=0.5)
            else:
                # Try without audio path (may fail if model needs conditionals)
                _ = model.generate(warmup_text, temperature=0.5, cfg_weight=0.5)
        except Exception as e:
            logger.warning(f"    Warmup 1 failed (may be expected): {e}")
            return 0.0
        warmup1_time = time.perf_counter() - start
        logger.info(f"    Time: {warmup1_time:.2f}s")
        
        # Second warmup - uses compiled code
        logger.info("  Warmup 2 (using compiled code)...")
        start = time.perf_counter()
        try:
            if warmup_audio_path:
                _ = model.generate(warmup_text, warmup_audio_path, temperature=0.5, cfg_weight=0.5)
            else:
                _ = model.generate(warmup_text, temperature=0.5, cfg_weight=0.5)
        except Exception as e:
            logger.warning(f"    Warmup 2 failed: {e}")
            return warmup1_time
        warmup2_time = time.perf_counter() - start
        logger.info(f"    Time: {warmup2_time:.2f}s")
        
        total_compilation_time = time.perf_counter() - compilation_start
        
        # Estimate compilation overhead
        compilation_overhead = warmup1_time - warmup2_time
        logger.info(f"  Estimated compilation overhead: {compilation_overhead:.2f}s")
        logger.info(f"  Total warmup time: {total_compilation_time:.2f}s")
        
        # Clear any GPU cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return total_compilation_time
    
    def get_optimization_summary(self, optimization_info: dict) -> str:
        """
        Get a human-readable summary of applied optimizations.
        
        Args:
            optimization_info: Dictionary with optimization details
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        if optimization_info.get('bfloat16'):
            summary_parts.append("BFloat16")
        
        if optimization_info.get('torch_compile'):
            mode = optimization_info.get('compile_mode', 'default')
            summary_parts.append(f"torch.compile({mode})")
        
        if optimization_info.get('reduced_cache'):
            summary_parts.append(f"cache={self.REDUCED_CACHE_LEN}")
        
        if summary_parts:
            return f"Optimizations: {', '.join(summary_parts)}"
        else:
            return "No optimizations applied"


def load_optimized_model(
    model_type: str,
    model_path: Optional[Path] = None,
    device: str = "cuda",
    perform_warmup: bool = True,
    warmup_audio_path: Optional[str] = None
) -> Tuple[ChatterboxTTS, dict]:
    """
    Convenience function to load and optimize a model in one step.
    
    Args:
        model_type: Type of model (base/grpo/quantized)
        model_path: Path to local model (for grpo/quantized)
        device: Device to load on
        perform_warmup: Whether to perform warmup compilation
        warmup_audio_path: Optional audio path for warmup
        
    Returns:
        Tuple of (optimized_model, optimization_info)
    """
    loader = OptimizedModelLoader(device)
    model, optimization_info = loader.load_and_optimize(model_type, model_path)
    
    if perform_warmup and optimization_info.get('torch_compile'):
        compilation_time = loader.warmup_model(model, warmup_audio_path=warmup_audio_path)
        optimization_info['compilation_time'] = compilation_time
    
    logger.info(loader.get_optimization_summary(optimization_info))
    
    return model, optimization_info
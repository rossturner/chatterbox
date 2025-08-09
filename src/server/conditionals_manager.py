import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.chatterbox.tts import Conditionals

logger = logging.getLogger(__name__)


class ConditionalsManager:
    """Manager for three-tier caching of voice conditionals"""
    
    def __init__(self, cache_dir: Path = Path("conditionals_cache")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.conditionals_ram: Dict[str, List[Tuple[Conditionals, str]]] = {}  # Store (conditionals, voice_sample_path)
        self.emotions_config: Dict[str, dict] = {}
        self.last_used_sample: Optional[str] = None
        
    def get_cache_path(self, emotion: str, audio_ref: str) -> Path:
        """Generate unique cache path for emotion/audio combination"""
        # Use full path in hash to ensure uniqueness
        ref_hash = hashlib.md5(audio_ref.encode()).hexdigest()[:8]
        return self.cache_dir / f"{emotion}_{ref_hash}.pt"
    
    def is_stale(self, cache_path: Path, audio_ref: Path) -> bool:
        """Check if cached conditional is older than audio file"""
        if not cache_path.exists():
            return True
        
        audio_path = Path(audio_ref)
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_ref}")
            return True
            
        # Check if audio file is newer than cache
        return audio_path.stat().st_mtime > cache_path.stat().st_mtime
    
    def prepare_conditionals(self, model, emotions_config: dict, batch_size: int = 5) -> Tuple[int, int]:
        """
        Prepare and cache all conditionals from emotions config.
        
        Args:
            model: ChatterboxTTS model instance
            emotions_config: Dictionary of emotion configurations
            batch_size: Number of conditionals to prepare at once
            
        Returns:
            Tuple of (prepared_count, loaded_count)
        """
        self.emotions_config = emotions_config
        prepared_count = 0
        loaded_count = 0
        
        # Phase 1: Check what needs to be prepared
        to_prepare = []
        conditionals_on_disk = {}
        
        for emotion, config in emotions_config.items():
            conditionals_on_disk[emotion] = []
            
            # Handle both dict and EmotionConfig objects
            voice_samples = config.voice_samples if hasattr(config, 'voice_samples') else config.get('voice_samples', [])
            exaggeration = config.exaggeration if hasattr(config, 'exaggeration') else config.get('exaggeration', 0.5)
            
            for voice_sample in voice_samples:
                cache_path = self.get_cache_path(emotion, voice_sample)
                
                if cache_path.exists() and not self.is_stale(cache_path, Path(voice_sample)):
                    # Conditional already on disk and up to date
                    conditionals_on_disk[emotion].append(cache_path)
                    loaded_count += 1
                    logger.info(f"Found cached conditional: {cache_path.name}")
                else:
                    # Need to prepare and save
                    to_prepare.append((emotion, voice_sample, cache_path, exaggeration))
                    
        # Phase 2: Prepare missing conditionals in batches
        if to_prepare:
            logger.info(f"Preparing {len(to_prepare)} missing conditionals in batches of {batch_size}")
            
            for i in range(0, len(to_prepare), batch_size):
                batch = to_prepare[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(to_prepare) + batch_size - 1)//batch_size}")
                
                for emotion, voice_sample, cache_path, exaggeration in batch:
                    try:
                        logger.info(f"Preparing conditional for {emotion} from {Path(voice_sample).name}")
                        model.prepare_conditionals(voice_sample, exaggeration)
                        model.conds.save(cache_path)
                        conditionals_on_disk[emotion].append(cache_path)
                        prepared_count += 1
                        logger.info(f"Saved conditional: {cache_path.name}")
                    except Exception as e:
                        logger.error(f"Failed to prepare conditional for {emotion} from {voice_sample}: {e}")
                
                # Clear VRAM between batches to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        # Phase 3: Load all conditionals into RAM
        logger.info("Loading all conditionals into RAM...")
        self.load_all_to_ram(conditionals_on_disk)
        
        return prepared_count, loaded_count
    
    def load_all_to_ram(self, conditionals_on_disk: Optional[Dict[str, List[Path]]] = None):
        """
        Load all cached conditionals into RAM on startup.
        
        Args:
            conditionals_on_disk: Optional dict of emotion -> list of cache paths
        """
        self.conditionals_ram.clear()
        
        if conditionals_on_disk:
            # Load specified conditionals
            for emotion, cache_paths in conditionals_on_disk.items():
                if emotion not in self.conditionals_ram:
                    self.conditionals_ram[emotion] = []
                
                for cache_path in cache_paths:
                    try:
                        # Load to CPU memory, not GPU
                        cond = Conditionals.load(cache_path, map_location="cpu")
                        
                        # Extract the voice sample path from the cache filename
                        # Cache files are named like: emotion_hashcode.pt
                        # We need to find which voice sample this corresponds to
                        voice_sample = self._find_voice_sample_for_cache(emotion, cache_path)
                        
                        self.conditionals_ram[emotion].append((cond, voice_sample))
                        logger.debug(f"Loaded {cache_path.name} to RAM for sample {voice_sample}")
                    except Exception as e:
                        logger.error(f"Failed to load conditional from {cache_path}: {e}")
        else:
            # Load all .pt files from cache directory
            for cache_file in self.cache_dir.glob("*.pt"):
                emotion = cache_file.stem.split("_")[0]
                if emotion not in self.conditionals_ram:
                    self.conditionals_ram[emotion] = []
                
                try:
                    cond = Conditionals.load(cache_file, map_location="cpu")
                    voice_sample = self._find_voice_sample_for_cache(emotion, cache_file)
                    self.conditionals_ram[emotion].append((cond, voice_sample))
                    logger.debug(f"Loaded {cache_file.name} to RAM")
                except Exception as e:
                    logger.error(f"Failed to load conditional from {cache_file}: {e}")
        
        # Report memory usage
        total_conditionals = sum(len(conds) for conds in self.conditionals_ram.values())
        estimated_ram_mb = total_conditionals * 0.11  # ~0.11 MB per conditional
        logger.info(f"Loaded {total_conditionals} conditionals into RAM (~{estimated_ram_mb:.1f} MB)")
        
        for emotion, conds in self.conditionals_ram.items():
            logger.info(f"  {emotion}: {len(conds)} conditionals")
    
    def _find_voice_sample_for_cache(self, emotion: str, cache_path: Path) -> str:
        """Find which voice sample a cache file corresponds to"""
        # Extract hash from cache filename
        cache_hash = cache_path.stem.split("_")[1] if "_" in cache_path.stem else ""
        
        # Check which voice sample produces this hash
        if emotion in self.emotions_config:
            config = self.emotions_config[emotion]
            voice_samples = config.voice_samples if hasattr(config, 'voice_samples') else config.get('voice_samples', [])
            
            for voice_sample in voice_samples:
                test_cache_path = self.get_cache_path(emotion, voice_sample)
                if test_cache_path.name == cache_path.name:
                    return voice_sample
        
        # Fallback: return the cache filename if we can't find the original
        return cache_path.stem
    
    def get_random_conditionals(self, emotion: str) -> Tuple[Conditionals, str]:
        """
        Get random conditionals for emotion from RAM.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Tuple of (Conditionals object, voice_sample_path)
            
        Raises:
            ValueError: If emotion not found
        """
        if emotion not in self.conditionals_ram:
            available = list(self.conditionals_ram.keys())
            raise ValueError(f"Unknown emotion: {emotion}. Available: {available}")
        
        conds_list = self.conditionals_ram[emotion]
        if not conds_list:
            raise ValueError(f"No conditionals available for emotion: {emotion}")
        
        cond, voice_sample = random.choice(conds_list)
        self.last_used_sample = voice_sample
        return cond, voice_sample
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions"""
        return list(self.conditionals_ram.keys())
    
    def get_emotion_config(self, emotion: str) -> Optional[dict]:
        """Get configuration for an emotion"""
        return self.emotions_config.get(emotion)
    
    def clear_cache(self):
        """Clear all cached conditionals from RAM"""
        self.conditionals_ram.clear()
        logger.info("Cleared conditionals from RAM")
        
    def delete_disk_cache(self):
        """Delete all cached conditionals from disk"""
        count = 0
        for cache_file in self.cache_dir.glob("*.pt"):
            cache_file.unlink()
            count += 1
        logger.info(f"Deleted {count} cached conditionals from disk")
    
    def clear_emotion_cache(self, emotion: str):
        """
        Clear all cached conditionals for a specific emotion.
        
        Args:
            emotion: Emotion name to clear cache for
        """
        logger.info(f"Clearing cache for emotion '{emotion}'")
        
        # Clear from RAM cache
        if emotion in self.conditionals_ram:
            del self.conditionals_ram[emotion]
            logger.info(f"Removed {emotion} from RAM cache")
        
        # Clear from disk cache
        count = 0
        for cache_file in self.cache_dir.glob(f"{emotion}_*.pt"):
            cache_file.unlink()
            count += 1
        
        if count > 0:
            logger.info(f"Deleted {count} cached conditionals for {emotion} from disk")
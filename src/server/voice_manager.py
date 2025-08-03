"""
Voice manager for handling emotion profiles and precomputed voice conditionals.
"""

import json
import uuid
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import torch
import librosa
import copy
import pickle

from chatterbox.tts import ChatterboxTTS, Conditionals
from .models import EmotionProfile, VoiceSample, EmotionProfileCreate, EmotionProfileUpdate
from .config import get_config

logger = logging.getLogger(__name__)


class VoiceManager:
    """Manages emotion profiles and precomputed voice conditionals."""
    
    def __init__(self, model: ChatterboxTTS):
        self.model = model
        self.config = get_config()
        self.emotions: Dict[str, EmotionProfile] = {}
        self.voice_conditionals: Dict[str, Conditionals] = {}
        self.voice_samples: Dict[str, VoiceSample] = {}
        self._load_lock = asyncio.Lock()
        
        # Create cache directory
        self.config.voice_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing emotions and conditionals
        asyncio.create_task(self._load_emotions())
    
    def _safe_copy_conditionals(self, conditionals: Conditionals) -> Conditionals:
        """Safely copy conditionals, handling potential deepcopy issues with compiled models."""
        try:
            # Try deepcopy first
            return copy.deepcopy(conditionals)
        except Exception:
            # Fallback: manual copy of the conditionals structure
            try:
                # Create new conditionals with copied tensor data
                new_t3_cond = conditionals.t3.__class__(
                    speaker_emb=conditionals.t3.speaker_emb.clone().detach(),
                    cond_prompt_speech_tokens=conditionals.t3.cond_prompt_speech_tokens.clone().detach() if conditionals.t3.cond_prompt_speech_tokens is not None else None,
                    emotion_adv=conditionals.t3.emotion_adv.clone().detach()
                )
                
                # Copy the generator dictionary with tensor cloning
                new_gen_dict = {}
                for k, v in conditionals.gen.items():
                    if torch.is_tensor(v):
                        new_gen_dict[k] = v.clone().detach()
                    else:
                        new_gen_dict[k] = v
                
                return Conditionals(new_t3_cond, new_gen_dict)
            except Exception as e:
                logger.error(f"Failed to copy conditionals safely: {e}")
                # Return original as last resort
                return conditionals
    
    async def _load_emotions(self):
        """Load emotion profiles from disk."""
        async with self._load_lock:
            try:
                if self.config.emotion_config_file.exists():
                    with open(self.config.emotion_config_file, 'r') as f:
                        emotions_data = json.load(f)
                    
                    for emotion_data in emotions_data.get('emotions', []):
                        emotion = EmotionProfile(**emotion_data)
                        self.emotions[emotion.id] = emotion
                        
                        # Load voice samples
                        for sample_path in emotion.voice_samples:
                            sample_id = f"{emotion.id}_{Path(sample_path).stem}"
                            if sample_id not in self.voice_samples:
                                sample = VoiceSample(
                                    id=sample_id,
                                    filename=Path(sample_path).name,
                                    file_path=sample_path
                                )
                                self.voice_samples[sample_id] = sample
                        
                        # Precompute conditionals for this emotion
                        await self._precompute_emotion_conditionals(emotion)
                
                logger.info(f"Loaded {len(self.emotions)} emotion profiles")
                
            except Exception as e:
                logger.error(f"Error loading emotions: {e}")
    
    async def _save_emotions(self):
        """Save emotion profiles to disk."""
        try:
            emotions_data = {
                'emotions': [emotion.dict() for emotion in self.emotions.values()],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config.emotion_config_file, 'w') as f:
                json.dump(emotions_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving emotions: {e}")
            raise
    
    async def _precompute_emotion_conditionals(self, emotion: EmotionProfile):
        """Precompute voice conditionals for an emotion profile."""
        try:
            # Check if cached conditionals exist
            cache_key = f"{emotion.id}_{emotion.exaggeration}_{hash(tuple(emotion.voice_samples))}"
            cache_path = self.config.get_cache_path(cache_key)
            
            if cache_path.exists():
                # Load from cache
                logger.info(f"Loading cached conditionals for emotion {emotion.name}")
                conditionals = torch.load(cache_path, map_location=self.model.device)
                self.voice_conditionals[emotion.id] = conditionals
                return
            
            # Find the primary voice sample (first one or longest one)
            if not emotion.voice_samples:
                logger.warning(f"No voice samples for emotion {emotion.name}")
                return
            
            primary_sample = None
            max_duration = 0
            
            for sample_path in emotion.voice_samples:
                full_path = self.config.get_voice_path(sample_path)
                if full_path.exists():
                    try:
                        # Quick duration check
                        y, sr = librosa.load(str(full_path), sr=None, duration=1.0)
                        duration = librosa.get_duration(y=y, sr=sr)
                        if duration > max_duration:
                            max_duration = duration
                            primary_sample = str(full_path)
                    except Exception as e:
                        logger.warning(f"Error checking sample {sample_path}: {e}")
                        continue
            
            if primary_sample is None:
                logger.error(f"No valid voice samples found for emotion {emotion.name}")
                return
            
            # Precompute conditionals using the primary sample
            logger.info(f"Precomputing conditionals for emotion {emotion.name} using {Path(primary_sample).name}")
            
            # Store original conditionals to restore later
            original_conds = self.model.conds
            
            try:
                # Prepare conditionals with the specified exaggeration
                self.model.prepare_conditionals(
                    primary_sample, 
                    exaggeration=emotion.exaggeration
                )
                
                # Store the computed conditionals (avoid deepcopy issues with compiled models)
                self.voice_conditionals[emotion.id] = self._safe_copy_conditionals(self.model.conds)
                
                # Cache to disk
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.model.conds, cache_path)
                
                logger.info(f"Successfully precomputed conditionals for emotion {emotion.name}")
                
            finally:
                # Restore original conditionals
                if original_conds is not None:
                    self.model.conds = original_conds
                
        except Exception as e:
            logger.error(f"Error precomputing conditionals for emotion {emotion.name}: {e}")
    
    async def create_emotion(self, emotion_data: EmotionProfileCreate) -> EmotionProfile:
        """Create a new emotion profile."""
        emotion_id = str(uuid.uuid4())
        
        emotion = EmotionProfile(
            id=emotion_id,
            name=emotion_data.name,
            character=emotion_data.character,
            voice_samples=[],  # Will be added separately
            exaggeration=emotion_data.exaggeration,
            description=emotion_data.description
        )
        
        self.emotions[emotion_id] = emotion
        await self._save_emotions()
        
        logger.info(f"Created emotion profile: {emotion.name} for character {emotion.character}")
        return emotion
    
    async def update_emotion(self, emotion_id: str, updates: EmotionProfileUpdate) -> Optional[EmotionProfile]:
        """Update an existing emotion profile."""
        if emotion_id not in self.emotions:
            return None
        
        emotion = self.emotions[emotion_id]
        update_data = updates.dict(exclude_unset=True)
        
        # Check if exaggeration changed - need to recompute conditionals
        exaggeration_changed = 'exaggeration' in update_data and update_data['exaggeration'] != emotion.exaggeration
        
        # Update emotion profile
        for field, value in update_data.items():
            setattr(emotion, field, value)
        
        emotion.updated_at = datetime.now()
        
        # Recompute conditionals if exaggeration changed
        if exaggeration_changed and emotion.voice_samples:
            await self._precompute_emotion_conditionals(emotion)
        
        await self._save_emotions()
        
        logger.info(f"Updated emotion profile: {emotion.name}")
        return emotion
    
    async def delete_emotion(self, emotion_id: str) -> bool:
        """Delete an emotion profile."""
        if emotion_id not in self.emotions:
            return False
        
        emotion = self.emotions[emotion_id]
        
        # Remove voice conditionals
        if emotion_id in self.voice_conditionals:
            del self.voice_conditionals[emotion_id]
        
        # Remove cached files
        cache_pattern = f"{emotion_id}_*"
        for cache_file in self.config.voice_cache_dir.glob(cache_pattern):
            cache_file.unlink()
        
        # Remove voice samples
        for sample_path in emotion.voice_samples:
            full_path = self.config.get_voice_path(sample_path)
            if full_path.exists():
                full_path.unlink()
        
        # Remove emotion profile
        del self.emotions[emotion_id]
        await self._save_emotions()
        
        logger.info(f"Deleted emotion profile: {emotion.name}")
        return True
    
    async def add_voice_sample(self, emotion_id: str, file_path: Path, filename: str, description: Optional[str] = None) -> Optional[VoiceSample]:
        """Add a voice sample to an emotion profile."""
        if emotion_id not in self.emotions:
            return None
        
        emotion = self.emotions[emotion_id]
        
        # Generate unique filename with shorter UUID prefix (remove extension from filename to avoid duplication)
        file_extension = file_path.suffix
        filename_without_ext = Path(filename).stem
        # Use only first 8 chars of emotion_id for shorter filenames
        short_emotion_id = emotion_id[:8] if len(emotion_id) > 8 else emotion_id
        sample_filename = f"{short_emotion_id}_{len(emotion.voice_samples)}_{filename_without_ext}{file_extension}"
        target_path = self.config.get_voice_path(sample_filename)
        
        # Copy file to storage
        shutil.copy2(file_path, target_path)
        
        # Create voice sample record
        sample_id = f"{emotion_id}_{Path(sample_filename).stem}"
        sample = VoiceSample(
            id=sample_id,
            filename=filename,
            file_path=sample_filename,
            description=description
        )
        
        # Add audio metadata
        try:
            y, sr = librosa.load(str(target_path), sr=None)
            sample.duration_seconds = librosa.get_duration(y=y, sr=sr)
            sample.sample_rate = sr
        except Exception as e:
            logger.warning(f"Could not extract audio metadata: {e}")
        
        # Update emotion profile
        emotion.voice_samples.append(sample_filename)
        emotion.updated_at = datetime.now()
        
        # Store sample reference
        self.voice_samples[sample_id] = sample
        
        # Recompute conditionals with new sample
        await self._precompute_emotion_conditionals(emotion)
        await self._save_emotions()
        
        logger.info(f"Added voice sample {filename} to emotion {emotion.name}")
        return sample
    
    async def remove_voice_sample(self, emotion_id: str, voice_filename: str) -> bool:
        """Remove a voice sample from an emotion profile."""
        if emotion_id not in self.emotions:
            return False
        
        emotion = self.emotions[emotion_id]
        
        # Find and remove the voice sample from the list
        logger.info(f"Looking for voice filename: '{voice_filename}' in samples: {emotion.voice_samples}")
        if voice_filename not in emotion.voice_samples:
            logger.warning(f"Voice filename '{voice_filename}' not found in emotion samples")
            return False
        
        emotion.voice_samples.remove(voice_filename)
        
        # Remove the physical file
        voice_path = self.config.get_voice_path(voice_filename)
        if voice_path.exists():
            voice_path.unlink()
        
        # Remove from voice samples dict
        sample_id = f"{emotion_id}_{Path(voice_filename).stem}"
        if sample_id in self.voice_samples:
            del self.voice_samples[sample_id]
        
        # Update emotion and save
        emotion.updated_at = datetime.now()
        
        # Recompute conditionals if there are still voice samples, otherwise clear them
        if emotion.voice_samples:
            await self._precompute_emotion_conditionals(emotion)
        else:
            # Clear conditionals if no voice samples left
            if emotion_id in self.voice_conditionals:
                del self.voice_conditionals[emotion_id]
        
        await self._save_emotions()
        
        logger.info(f"Removed voice sample {voice_filename} from emotion {emotion.name}")
        return True
    
    def get_emotion(self, emotion_id: str) -> Optional[EmotionProfile]:
        """Get an emotion profile by ID."""
        return self.emotions.get(emotion_id)
    
    def list_emotions(self) -> List[EmotionProfile]:
        """List all emotion profiles."""
        return list(self.emotions.values())
    
    def list_characters(self) -> List[str]:
        """List all unique character names."""
        characters = set()
        for emotion in self.emotions.values():
            characters.add(emotion.character)
        return sorted(list(characters))
    
    def get_emotions_by_character(self, character: str) -> List[EmotionProfile]:
        """Get all emotions for a specific character."""
        return [emotion for emotion in self.emotions.values() if emotion.character == character]
    
    async def use_emotion(self, emotion_id: str) -> bool:
        """Switch the model to use a specific emotion's conditionals."""
        if emotion_id not in self.voice_conditionals:
            logger.error(f"No precomputed conditionals for emotion {emotion_id}")
            return False
        
        # Switch to the emotion's conditionals
        self.model.conds = self.voice_conditionals[emotion_id]
        logger.debug(f"Switched to emotion {emotion_id}")
        return True
    
    def get_emotion_conditionals(self, emotion_id: str) -> Optional[Conditionals]:
        """Get precomputed conditionals for an emotion."""
        return self.voice_conditionals.get(emotion_id)
    
    def is_emotion_ready(self, emotion_id: str) -> bool:
        """Check if an emotion has precomputed conditionals ready."""
        return emotion_id in self.voice_conditionals
    
    def get_stats(self) -> Dict[str, Any]:
        """Get voice manager statistics."""
        total_samples = sum(len(emotion.voice_samples) for emotion in self.emotions.values())
        ready_emotions = len(self.voice_conditionals)
        
        return {
            'total_emotions': len(self.emotions),
            'ready_emotions': ready_emotions,
            'total_voice_samples': total_samples,
            'unique_characters': len(self.list_characters()),
            'cache_files': len(list(self.config.voice_cache_dir.glob('*.pt')))
        }
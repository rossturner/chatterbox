"""
Emotion manager for handling emotion profiles with persistence.
"""

import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import yaml

from .models import EmotionConfig, EmotionCreateRequest, EmotionUpdateRequest, VoiceSample
from .conditionals_manager import ConditionalsManager
from .config import ServerConfig

logger = logging.getLogger(__name__)


class EmotionManager:
    """Manages emotion profiles with CRUD operations and persistence."""
    
    def __init__(self, config: ServerConfig, conditionals_manager: ConditionalsManager):
        self.config = config
        self.conditionals_manager = conditionals_manager
        self.emotions: Dict[str, EmotionConfig] = {}
        self.voice_samples_dir = Path("configs/voice_samples")
        self.voice_samples_dir.mkdir(parents=True, exist_ok=True)
        self.emotions_file = Path("configs/emotions.yaml")  # Use fixed path
        
        # Load existing emotions
        self._load_emotions()
    
    def _load_emotions(self):
        """Load emotions from YAML file."""
        try:
            if self.emotions_file.exists():
                with open(self.emotions_file, 'r') as f:
                    data = yaml.safe_load(f)
                    
                if data and 'emotions' in data:
                    for emotion_id, config in data['emotions'].items():
                        # Convert to EmotionConfig model
                        if 'created_at' not in config:
                            config['created_at'] = datetime.utcnow().isoformat()
                        if 'updated_at' not in config:
                            config['updated_at'] = config['created_at']
                        if 'name' not in config:
                            config['name'] = emotion_id
                            
                        self.emotions[emotion_id] = EmotionConfig(**config)
                        
                logger.info(f"Loaded {len(self.emotions)} emotions from {self.emotions_file}")
                
        except Exception as e:
            logger.error(f"Error loading emotions: {e}")
    
    def _save_emotions(self):
        """Save emotions to YAML file."""
        try:
            
            # Prepare data for YAML
            data = {'emotions': {}}
            for emotion_id, config in self.emotions.items():
                data['emotions'][emotion_id] = {
                    'name': config.name,
                    'exaggeration': config.exaggeration,
                    'voice_samples': config.voice_samples,
                    'created_at': config.created_at,
                    'updated_at': config.updated_at
                }
            
            # Save to YAML
            with open(self.emotions_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Saved {len(self.emotions)} emotions to {self.emotions_file}")
            
        except Exception as e:
            logger.error(f"Error saving emotions: {e}")
            raise
    
    def list_emotions(self) -> Dict[str, EmotionConfig]:
        """List all emotions."""
        return self.emotions
    
    def get_emotion(self, emotion_id: str) -> Optional[EmotionConfig]:
        """Get a specific emotion by ID."""
        return self.emotions.get(emotion_id)
    
    def create_emotion(self, emotion_id: str, request: EmotionCreateRequest) -> EmotionConfig:
        """Create a new emotion."""
        if emotion_id in self.emotions:
            raise ValueError(f"Emotion '{emotion_id}' already exists")
        
        # Create emotion config
        now = datetime.utcnow().isoformat()
        emotion = EmotionConfig(
            name=request.name,
            exaggeration=request.exaggeration,
            voice_samples=request.voice_samples or [],
            created_at=now,
            updated_at=now
        )
        
        # Save emotion
        self.emotions[emotion_id] = emotion
        self._save_emotions()
        
        # Clear cached conditionals for this emotion
        self.conditionals_manager.clear_emotion_cache(emotion_id)
        
        logger.info(f"Created emotion '{emotion_id}'")
        return emotion
    
    def update_emotion(self, emotion_id: str, request: EmotionUpdateRequest) -> EmotionConfig:
        """Update an existing emotion."""
        if emotion_id not in self.emotions:
            raise ValueError(f"Emotion '{emotion_id}' not found")
        
        emotion = self.emotions[emotion_id]
        updated = False
        
        # Update fields if provided
        if request.name is not None:
            emotion.name = request.name
            updated = True
        
        if request.exaggeration is not None:
            emotion.exaggeration = request.exaggeration
            updated = True
            # Clear cache when exaggeration changes
            self.conditionals_manager.clear_emotion_cache(emotion_id)
        
        if updated:
            emotion.updated_at = datetime.utcnow().isoformat()
            self._save_emotions()
            logger.info(f"Updated emotion '{emotion_id}'")
        
        return emotion
    
    def delete_emotion(self, emotion_id: str) -> bool:
        """Delete an emotion."""
        if emotion_id not in self.emotions:
            raise ValueError(f"Emotion '{emotion_id}' not found")
        
        # Remove emotion
        del self.emotions[emotion_id]
        self._save_emotions()
        
        # Clear cached conditionals
        self.conditionals_manager.clear_emotion_cache(emotion_id)
        
        # Delete associated voice samples directory if it exists
        emotion_samples_dir = self.voice_samples_dir / emotion_id
        if emotion_samples_dir.exists():
            shutil.rmtree(emotion_samples_dir)
            logger.info(f"Deleted voice samples directory for '{emotion_id}'")
        
        logger.info(f"Deleted emotion '{emotion_id}'")
        return True
    
    def add_voice_sample(self, emotion_id: str, sample_path: str) -> EmotionConfig:
        """Add a voice sample to an emotion."""
        if emotion_id not in self.emotions:
            raise ValueError(f"Emotion '{emotion_id}' not found")
        
        emotion = self.emotions[emotion_id]
        
        # Check if sample already exists
        if sample_path not in emotion.voice_samples:
            emotion.voice_samples.append(sample_path)
            emotion.updated_at = datetime.utcnow().isoformat()
            self._save_emotions()
            
            # Clear cached conditionals when voice samples change
            self.conditionals_manager.clear_emotion_cache(emotion_id)
            
            logger.info(f"Added voice sample '{sample_path}' to emotion '{emotion_id}'")
        
        return emotion
    
    def remove_voice_sample(self, emotion_id: str, sample_path: str) -> EmotionConfig:
        """Remove a voice sample from an emotion."""
        if emotion_id not in self.emotions:
            raise ValueError(f"Emotion '{emotion_id}' not found")
        
        emotion = self.emotions[emotion_id]
        
        # Remove sample if it exists
        if sample_path in emotion.voice_samples:
            emotion.voice_samples.remove(sample_path)
            emotion.updated_at = datetime.utcnow().isoformat()
            self._save_emotions()
            
            # Clear cached conditionals when voice samples change
            self.conditionals_manager.clear_emotion_cache(emotion_id)
            
            # Delete the actual file if it's in our managed directory
            sample_file = Path(sample_path)
            if not sample_file.is_absolute():
                sample_file = Path.cwd() / sample_file
            
            # Check if the file is within our voice samples directory for safety
            voice_samples_abs = (Path.cwd() / "configs" / "voice_samples").resolve()
            try:
                sample_file_resolved = sample_file.resolve()
                if sample_file_resolved.is_relative_to(voice_samples_abs):
                    if sample_file.exists():
                        sample_file.unlink()
                        logger.info(f"Deleted voice sample file '{sample_path}'")
                    else:
                        logger.warning(f"Voice sample file not found: '{sample_path}'")
                else:
                    logger.warning(f"Voice sample file outside managed directory, not deleting: '{sample_path}'")
            except (OSError, ValueError) as e:
                logger.error(f"Error resolving path for deletion: '{sample_path}': {e}")
            
            logger.info(f"Removed voice sample '{sample_path}' from emotion '{emotion_id}'")
        
        return emotion
    
    def list_voice_samples(self, emotion_id: str) -> List[VoiceSample]:
        """List all voice samples for an emotion."""
        if emotion_id not in self.emotions:
            raise ValueError(f"Emotion '{emotion_id}' not found")
        
        emotion = self.emotions[emotion_id]
        samples = []
        
        for sample_path in emotion.voice_samples:
            sample_file = Path(sample_path)
            if sample_file.exists():
                stat = sample_file.stat()
                samples.append(VoiceSample(
                    filename=sample_file.name,
                    path=sample_path,
                    duration=None,  # Could calculate this if needed
                    uploaded_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
                ))
        
        return samples
    
    def save_uploaded_voice_sample(self, emotion_id: str, filename: str, content: bytes) -> str:
        """Save an uploaded voice sample file."""
        # Create emotion-specific directory
        emotion_samples_dir = self.voice_samples_dir / emotion_id
        emotion_samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename if needed
        sample_path = emotion_samples_dir / filename
        if sample_path.exists():
            base = sample_path.stem
            ext = sample_path.suffix
            counter = 1
            while sample_path.exists():
                sample_path = emotion_samples_dir / f"{base}_{counter}{ext}"
                counter += 1
        
        # Save file
        with open(sample_path, 'wb') as f:
            f.write(content)
        
        # Return relative path from project root
        try:
            relative_path = str(sample_path.relative_to(Path.cwd()))
            logger.info(f"Generated relative path: {relative_path}")
            return relative_path
        except ValueError as e:
            logger.error(f"Error generating relative path for {sample_path} relative to {Path.cwd()}: {e}")
            # Fall back to relative path from expected project root instead of absolute path
            # Assume we're in the project root and construct relative path manually
            project_root = Path.cwd()
            if sample_path.is_absolute():
                # Try to find common path components to construct relative path
                parts = sample_path.parts
                if 'configs' in parts:
                    # Find configs directory and construct relative path from there
                    configs_index = parts.index('configs')
                    relative_parts = parts[configs_index:]
                    relative_path = str(Path(*relative_parts))
                elif 'audio_data' in parts:
                    # Find audio_data directory and construct relative path from there
                    audio_data_index = parts.index('audio_data')
                    relative_parts = parts[audio_data_index:]
                    relative_path = str(Path(*relative_parts))
                else:
                    # Last resort: use filename only (not ideal but better than absolute path)
                    relative_path = sample_path.name
            else:
                # Already relative, use as-is
                relative_path = str(sample_path)
            
            logger.info(f"Generated fallback relative path: {relative_path}")
            return relative_path
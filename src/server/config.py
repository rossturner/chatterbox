"""
Configuration management for the Chatterbox TTS API server.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import torch


class ServerConfig(BaseSettings):
    """Server configuration settings."""
    
    # Server settings
    host: str = Field(default="127.0.0.1", env="CHATTERBOX_HOST")
    port: int = Field(default=8000, env="CHATTERBOX_PORT")
    workers: int = Field(default=1, env="CHATTERBOX_WORKERS")
    reload: bool = Field(default=False, env="CHATTERBOX_RELOAD")
    
    # API settings
    api_title: str = "Chatterbox TTS API"
    api_version: str = "0.1.0"
    api_description: str = "High-performance Text-to-Speech API with voice cloning and emotion control"
    
    # Storage paths
    storage_root: Path = Field(default=Path("src/server/storage"))
    voices_dir: Optional[Path] = Field(default=None)
    configs_dir: Optional[Path] = Field(default=None) 
    cache_dir: Optional[Path] = Field(default=None)
    outputs_dir: Optional[Path] = Field(default=None)
    
    # Model settings
    model_device: Optional[str] = Field(default=None, env="CHATTERBOX_DEVICE")
    enable_torch_compile: bool = Field(default=True, env="CHATTERBOX_TORCH_COMPILE")
    compile_mode: str = Field(default="reduce-overhead", env="CHATTERBOX_COMPILE_MODE")
    enable_mixed_precision: bool = Field(default=True, env="CHATTERBOX_MIXED_PRECISION")
    
    # Performance settings
    max_text_length: int = Field(default=2000, env="CHATTERBOX_MAX_TEXT_LENGTH")
    max_generation_time: int = Field(default=120, env="CHATTERBOX_MAX_GENERATION_TIME")  # seconds
    cleanup_interval: int = Field(default=300, env="CHATTERBOX_CLEANUP_INTERVAL")  # seconds
    max_cached_outputs: int = Field(default=100, env="CHATTERBOX_MAX_CACHED_OUTPUTS")
    
    # Voice management
    max_voice_file_size: int = Field(default=50 * 1024 * 1024, env="CHATTERBOX_MAX_VOICE_SIZE")  # 50MB
    allowed_audio_formats: list = Field(default=["wav", "mp3", "flac", "ogg"])
    voice_sample_rate: int = Field(default=24000)
    
    # Security settings
    max_upload_size: int = Field(default=50 * 1024 * 1024)  # 50MB
    cors_origins: list = Field(default=["*"])
    enable_docs: bool = Field(default=True, env="CHATTERBOX_ENABLE_DOCS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_paths()
        self._setup_device()
    
    def _setup_paths(self):
        """Initialize storage paths."""
        self.storage_root = Path(self.storage_root).resolve()
        
        if self.voices_dir is None:
            self.voices_dir = self.storage_root / "voices"
        if self.configs_dir is None:
            self.configs_dir = self.storage_root / "configs"
        if self.cache_dir is None:
            self.cache_dir = self.storage_root / "cache"
        if self.outputs_dir is None:
            self.outputs_dir = self.storage_root / "outputs"
        
        # Create directories if they don't exist
        for path in [self.voices_dir, self.configs_dir, self.cache_dir, self.outputs_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self):
        """Auto-detect device if not specified."""
        if self.model_device is None:
            if torch.cuda.is_available():
                self.model_device = "cuda"
            elif torch.backends.mps.is_available():
                self.model_device = "mps"
            else:
                self.model_device = "cpu"
    
    @property
    def emotion_config_file(self) -> Path:
        """Path to the emotion profiles configuration file."""
        return self.configs_dir / "emotions.json"
    
    @property
    def voice_cache_dir(self) -> Path:
        """Directory for cached voice conditionals."""
        return self.cache_dir / "voice_conditionals"
    
    def get_voice_path(self, filename: str) -> Path:
        """Get full path for a voice file."""
        return self.voices_dir / filename
    
    def get_output_path(self, filename: str) -> Path:
        """Get full path for an output file."""
        return self.outputs_dir / filename
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get full path for a cache file."""
        return self.voice_cache_dir / f"{cache_key}.pt"


# Global configuration instance
config = ServerConfig()


def get_config() -> ServerConfig:
    """Get the current server configuration."""
    return config


def update_config(**kwargs) -> ServerConfig:
    """Update server configuration."""
    global config
    config = ServerConfig(**{**config.dict(), **kwargs})
    return config
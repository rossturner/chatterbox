import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    request_timeout: float = 30.0
    busy_timeout: float = 1.0


@dataclass
class ModelConfig:
    """Model configuration"""
    type: str = "base"  # "base", "grpo", or "quantized"
    path: Optional[str] = None
    device: str = "cuda"
    keep_warm: bool = True
    use_optimizations: bool = True  # Enable torch.compile and other optimizations


@dataclass
class GenerationConfig:
    """Generation configuration"""
    default_temperature: float = 0.8
    default_cfg_weight: float = 0.5
    max_text_length: int = 500
    use_cuda_graphs: bool = False


@dataclass
class CachingConfig:
    """Caching configuration"""
    precompute_on_startup: bool = True
    save_conditionals: bool = True
    conditionals_dir: str = "./conditionals_cache"
    batch_size: int = 5


@dataclass
class StreamingConfig:
    """Streaming configuration"""
    enabled: bool = False
    default_chunk_size: int = 25
    default_context_window: int = 50
    fade_duration: float = 0.02


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    enabled: bool = True
    max_connections: int = 10
    connection_timeout: float = 30.0
    max_request_size: int = 1000
    include_progress: bool = True
    sentence_streaming: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    pin_memory: bool = True
    torch_compile: bool = False
    single_request_mode: bool = True


@dataclass
class Config:
    """Complete server configuration"""
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        config = cls()
        
        # Parse server config
        if 'server' in data:
            config.server = ServerConfig(**data['server'])
        
        # Parse model config
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        
        # Parse generation config
        if 'generation' in data:
            config.generation = GenerationConfig(**data['generation'])
        
        # Parse caching config
        if 'caching' in data:
            config.caching = CachingConfig(**data['caching'])
        
        # Parse streaming config
        if 'streaming' in data:
            config.streaming = StreamingConfig(**data['streaming'])
        
        # Parse websocket config
        if 'websocket' in data:
            config.websocket = WebSocketConfig(**data['websocket'])
        
        # Parse performance config
        if 'performance' in data:
            config.performance = PerformanceConfig(**data['performance'])
        
        return config
    
    def validate(self):
        """Validate configuration"""
        # Ensure single worker for single model instance
        if self.server.workers != 1:
            logger.warning("Forcing workers=1 for single model instance")
            self.server.workers = 1
        
        # Validate model type
        valid_types = ["base", "grpo", "quantized"]
        if self.model.type not in valid_types:
            raise ValueError(f"Invalid model type: {self.model.type}. Must be one of {valid_types}")
        
        # Check if local model path exists
        if self.model.type != "base" and self.model.path:
            model_path = Path(self.model.path)
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {self.model.path}")


@dataclass
class EmotionConfig:
    """Configuration for a single emotion"""
    exaggeration: float
    voice_samples: list


def load_emotions_config(path: Path) -> Dict[str, EmotionConfig]:
    """Load emotions configuration from YAML file"""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    emotions = {}
    for name, config in data.get('emotions', {}).items():
        emotions[name] = EmotionConfig(
            exaggeration=config.get('exaggeration', 0.5),
            voice_samples=config.get('voice_samples', [])
        )
        
        # Validate voice samples exist
        for sample in emotions[name].voice_samples:
            sample_path = Path(sample)
            if not sample_path.exists():
                logger.warning(f"Voice sample not found: {sample}")
    
    return emotions
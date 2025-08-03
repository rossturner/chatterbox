#!/usr/bin/env python3
"""
Startup script for the Chatterbox TTS API server.
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.server.server import run_server
from src.server.config import get_config


def setup_example_emotions():
    """Set up example emotion profiles using existing test audio files."""
    config = get_config()
    
    # Check if we have the test audio files
    test_dir = project_root / "test"
    nicole_files = [
        "Galgame_ReliabilityNicole_Nicole_002.wav",
        "Galgame_Chapter0_Nicole_18_04.wav", 
        "Galgame_GoldenWeek_Nicole_19.wav"
    ]
    
    existing_files = []
    for file in nicole_files:
        file_path = test_dir / file
        if file_path.exists():
            existing_files.append(file.replace("Galgame_", "").replace("_Nicole", ""))
    
    if not existing_files:
        print("No test audio files found. You can add emotion profiles through the web interface.")
        return
    
    # Copy test files to server storage
    print("Setting up example emotion profiles...")
    for i, original_file in enumerate(nicole_files):
        source_path = test_dir / original_file
        if source_path.exists():
            target_filename = f"nicole_sample_{i+1}.wav"
            target_path = config.get_voice_path(target_filename)
            
            if not target_path.exists():
                import shutil
                shutil.copy2(source_path, target_path)
                print(f"Copied {original_file} -> {target_filename}")
    
    # Create example emotions config
    emotions_config = {
        "emotions": [
            {
                "id": "nicole_neutral",
                "name": "Neutral",
                "character": "Nicole", 
                "voice_samples": ["nicole_sample_1.wav"],
                "exaggeration": 0.3,
                "description": "Nicole's neutral speaking voice",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            },
            {
                "id": "nicole_expressive", 
                "name": "Expressive",
                "character": "Nicole",
                "voice_samples": ["nicole_sample_2.wav"],
                "exaggeration": 0.7,
                "description": "Nicole's more expressive emotional voice",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            },
            {
                "id": "nicole_dramatic",
                "name": "Dramatic", 
                "character": "Nicole",
                "voice_samples": ["nicole_sample_3.wav"],
                "exaggeration": 0.9,
                "description": "Nicole's dramatic and emphasized voice",
                "created_at": "2024-01-01T00:00:00", 
                "updated_at": "2024-01-01T00:00:00"
            }
        ],
        "last_updated": "2024-01-01T00:00:00"
    }
    
    # Only create if config doesn't exist
    if not config.emotion_config_file.exists():
        with open(config.emotion_config_file, 'w') as f:
            json.dump(emotions_config, f, indent=2)
        print(f"Created example emotions config with {len(emotions_config['emotions'])} emotions")
    else:
        print("Emotions config already exists, skipping setup")


def main():
    """Main entry point."""
    print("🎙️ Chatterbox TTS API Server")
    print("=" * 50)
    
    config = get_config()
    print(f"Device: {config.model_device}")
    print(f"Host: {config.host}:{config.port}")
    print(f"Storage: {config.storage_root}")
    
    # Setup example emotions if needed
    setup_example_emotions()
    
    print("\nStarting server...")
    print(f"Web interface will be available at: http://{config.host}:{config.port}")
    print(f"API documentation at: http://{config.host}:{config.port}/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the server
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\nServer error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
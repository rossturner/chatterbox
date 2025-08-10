"""
GRPO V3 Dataset - Audio dataset handling and loading
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import librosa
from tqdm import tqdm

from .config import *
from .utils import normalize_audio_sample_rate


@dataclass
class AudioSample:
    audio_path: str
    transcript: str
    duration: float
    sample_rate: int


def load_audio_samples_from_pairs(audio_dir: str) -> List[AudioSample]:
    """Load audio files and transcripts from paired .wav/.txt files with sample rate tracking"""
    samples = []
    audio_dir = Path(audio_dir)
    
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")
    
    sample_rates = {}
    
    for wav_path in tqdm(wav_files, desc="Processing paired files"):
        txt_path = wav_path.with_suffix('.txt')
        
        if not txt_path.exists():
            print(f"Warning: No matching .txt file for {wav_path.name}")
            continue
        
        try:
            # Load transcript
            with open(txt_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            if not transcript:
                print(f"Warning: Empty transcript in {txt_path.name}")
                continue
            
            # Get audio info without loading the full audio
            try:
                import soundfile as sf
                info = sf.info(str(wav_path))
                duration = info.duration
                sample_rate = info.samplerate
            except:
                # Fallback to librosa if soundfile fails
                y, sample_rate = librosa.load(str(wav_path), sr=None)
                duration = len(y) / sample_rate
            
            # Track sample rates
            if sample_rate not in sample_rates:
                sample_rates[sample_rate] = 0
            sample_rates[sample_rate] += 1
            
            # Filter by duration
            if MIN_AUDIO_LENGTH <= duration <= MAX_AUDIO_LENGTH:
                samples.append(AudioSample(
                    audio_path=str(wav_path),
                    transcript=transcript,
                    duration=duration,
                    sample_rate=int(sample_rate)
                ))
            else:
                print(f"Filtered out {wav_path.name}: duration {duration:.2f}s outside [{MIN_AUDIO_LENGTH}, {MAX_AUDIO_LENGTH}]s")
                
        except Exception as e:
            print(f"Error processing {wav_path.name}: {e}")
            continue
    
    print(f"Successfully loaded {len(samples)} paired samples")
    
    # Display sample rate distribution
    print("\nSample Rate Distribution:")
    for sr, count in sorted(sample_rates.items()):
        percentage = (count / len(wav_files)) * 100
        resample_note = " (will resample)" if sr != TARGET_SAMPLE_RATE else " (native)"
        print(f"  {sr}Hz: {count} files ({percentage:.1f}%){resample_note}")
    
    return samples


class PairedAudioDataset(Dataset):
    def __init__(self, samples: List[AudioSample], model):
        self.samples = samples
        self.model = model
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load and normalize audio
            audio, orig_sr = librosa.load(sample.audio_path, sr=None)
            
            # Normalize to target sample rate (48kHz)
            if orig_sr != TARGET_SAMPLE_RATE:
                audio = normalize_audio_sample_rate(audio, orig_sr, TARGET_SAMPLE_RATE)
            
            # Generate speaker embedding
            try:
                # Resample for voice encoder (16kHz)
                audio_16k = librosa.resample(audio, orig_sr=TARGET_SAMPLE_RATE, target_sr=16000)
                speaker_emb = self.model.ve.embeds_from_wavs([audio_16k], sample_rate=16000)
                
                if isinstance(speaker_emb, torch.Tensor):
                    speaker_emb = speaker_emb.squeeze()
                elif isinstance(speaker_emb, (list, tuple)):
                    speaker_emb = torch.tensor(speaker_emb[0]) if len(speaker_emb) > 0 else torch.zeros(256)
                else:
                    speaker_emb = torch.tensor(speaker_emb)
                    
                # Ensure proper shape
                if speaker_emb.dim() == 0:
                    speaker_emb = torch.zeros(256)
                elif speaker_emb.dim() > 1:
                    speaker_emb = speaker_emb.flatten()
                    
            except Exception as e:
                print(f"Error generating speaker embedding for {sample.audio_path}: {e}")
                speaker_emb = torch.zeros(256)
            
            return {
                "text": sample.transcript,
                "audio": torch.tensor(audio, dtype=torch.float32),
                "speaker_emb": speaker_emb.float(),
                "duration": torch.tensor([sample.duration], dtype=torch.float32),
                "sample_rate": sample.sample_rate,
                "path": sample.audio_path
            }
            
        except Exception as e:
            print(f"Error loading sample {sample.audio_path}: {e}")
            # Return None to signal this sample should be skipped
            return None
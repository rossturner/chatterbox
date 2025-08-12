#!/usr/bin/env python3

import os
import wave
import numpy as np
import shutil
from pathlib import Path
import scipy.signal

def get_audio_duration(filepath):
    """Get duration of audio file in seconds."""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            return frames / sample_rate
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def read_transcript(filepath):
    """Read transcript text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading transcript {filepath}: {e}")
        return ""

def concatenate_audio(input_files, output_file):
    """Concatenate multiple WAV files into one with silence gaps, resampling as needed."""
    if not input_files:
        return False
    
    try:
        target_sample_rate = 48000  # Standard sample rate
        target_sample_width = 2
        target_channels = 1
        silence_duration = 0.4  # 400ms silence gap between clips
        silence_samples = int(silence_duration * target_sample_rate)
        silence_gap = np.zeros(silence_samples, dtype=np.float32)
        
        all_audio_data = []
        
        # Read and resample all files
        for i, input_file in enumerate(input_files):
            with wave.open(input_file, 'rb') as input_wav:
                sample_rate = input_wav.getframerate()
                sample_width = input_wav.getsampwidth()
                channels = input_wav.getnchannels()
                frames = input_wav.readframes(input_wav.getnframes())
                
                # Convert to numpy array
                if sample_width == 1:
                    audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128
                elif sample_width == 2:
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                elif sample_width == 4:
                    audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
                else:
                    print(f"Unsupported sample width: {sample_width}")
                    continue
                
                # Handle multi-channel audio by taking first channel
                if channels > 1:
                    audio_data = audio_data.reshape(-1, channels)[:, 0]
                
                # Resample if needed
                if sample_rate != target_sample_rate:
                    num_samples = int(len(audio_data) * target_sample_rate / sample_rate)
                    audio_data = scipy.signal.resample(audio_data, num_samples)
                
                all_audio_data.append(audio_data)
                
                # Add silence gap between clips (except after the last clip)
                if i < len(input_files) - 1:
                    all_audio_data.append(silence_gap)
        
        # Concatenate all audio
        if all_audio_data:
            combined_audio = np.concatenate(all_audio_data)
            
            # Convert back to int16
            combined_audio = np.clip(combined_audio, -32768, 32767).astype(np.int16)
            
            # Write output file
            with wave.open(output_file, 'wb') as output_wav:
                output_wav.setnchannels(target_channels)
                output_wav.setsampwidth(target_sample_width)
                output_wav.setframerate(target_sample_rate)
                output_wav.writeframes(combined_audio.tobytes())
        
        return True
    except Exception as e:
        print(f"Error concatenating audio: {e}")
        return False

def combine_audio_clips():
    """Main function to combine audio clips."""
    
    # Setup directories
    input_dir = Path("./audio_data_v2")
    output_dir = Path("./audio_data_v3")
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Get all audio files and their durations
    audio_info = []
    for filepath in input_dir.glob("*.wav"):
        duration = get_audio_duration(str(filepath))
        if duration > 0:
            audio_info.append({
                'wav_file': filepath,
                'txt_file': filepath.with_suffix('.txt'),
                'duration': duration,
                'used': False
            })
    
    # Sort by duration descending
    audio_info.sort(key=lambda x: x['duration'], reverse=True)
    
    print(f"Found {len(audio_info)} audio files")
    
    combined_count = 0
    copied_count = 0
    
    # First pass: copy files >= 10 seconds
    for info in audio_info:
        if info['duration'] >= 10.0:
            # Copy both wav and txt files
            wav_dest = output_dir / info['wav_file'].name
            txt_dest = output_dir / info['txt_file'].name
            
            shutil.copy2(info['wav_file'], wav_dest)
            if info['txt_file'].exists():
                shutil.copy2(info['txt_file'], txt_dest)
            
            info['used'] = True
            copied_count += 1
            print(f"Copied long clip: {info['wav_file'].name} ({info['duration']:.2f}s)")
    
    # Second pass: combine short clips
    unused_clips = [info for info in audio_info if not info['used']]
    unused_clips.sort(key=lambda x: x['duration'], reverse=True)  # Process longest unused first
    
    while unused_clips:
        # Take the longest unused clip as the base
        base_clip = unused_clips.pop(0)
        combination = [base_clip]
        total_duration = base_clip['duration']
        
        # Find clips to combine with it to reach >= 10 seconds
        unused_clips.sort(key=lambda x: x['duration'])  # Sort by shortest first for greedy selection
        
        # Keep adding clips until we reach at least 10 seconds
        while total_duration < 10.0 and unused_clips:
            # Find the best clip to add (closest to reaching 10s without going over 25s)
            best_clip = None
            best_index = -1
            
            for i, clip in enumerate(unused_clips):
                new_total = total_duration + clip['duration']
                if new_total <= 25.0:  # Don't exceed 25 seconds
                    if best_clip is None or clip['duration'] > best_clip['duration']:
                        best_clip = clip
                        best_index = i
            
            if best_clip is None:
                # No suitable clip found, break
                break
                
            combination.append(best_clip)
            total_duration += best_clip['duration']
            unused_clips.pop(best_index)
        
        # Skip combinations that are still too short
        if total_duration < 10.0:
            print(f"Warning: Could not reach 10s for combination starting with {base_clip['wav_file'].name} ({total_duration:.2f}s)")
            continue
        
        # Sort combination by original filename order for consistent transcript ordering
        combination.sort(key=lambda x: x['wav_file'].name)
        
        # Create combined files
        combined_count += 1
        output_wav = output_dir / f"combined_{combined_count:03d}.wav"
        output_txt = output_dir / f"combined_{combined_count:03d}.txt"
        
        # Combine audio files
        wav_files = [str(clip['wav_file']) for clip in combination]
        if concatenate_audio(wav_files, str(output_wav)):
            # Combine transcripts
            transcripts = []
            for clip in combination:
                if clip['txt_file'].exists():
                    transcript = read_transcript(str(clip['txt_file']))
                    if transcript:
                        transcripts.append(transcript)
            
            combined_transcript = " ".join(transcripts)
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(combined_transcript)
            
            clip_names = [clip['wav_file'].name for clip in combination]
            print(f"Combined {len(combination)} clips into {output_wav.name} ({total_duration:.2f}s): {', '.join(clip_names)}")
        else:
            print(f"Failed to combine clips for {output_wav.name}")
    
    print(f"\nSummary:")
    print(f"- Copied {copied_count} long clips (>=10s)")
    print(f"- Created {combined_count} combined clips")
    print(f"- Total output files: {copied_count + combined_count}")

if __name__ == "__main__":
    combine_audio_clips()
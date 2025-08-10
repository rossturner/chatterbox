#!/usr/bin/env python3
"""
WebSocket Performance Test Harness for Chatterbox Streaming TTS

This script tests the WebSocket-based sentence streaming implementation,
measuring time-to-first-sentence (TTFS), overall generation performance,
and comparing against the existing HTTP endpoint performance.

Modified to produce a single, longer output made up of 2 transcriptions
taken from ./audio_data/transcripts_cache.json
"""

import asyncio
import json
import time
import base64
import io
import uuid
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

import websockets
import torch
import torchaudio
import requests
from tabulate import tabulate

# Test configuration
SERVER_HOST = "localhost"
SERVER_PORT = 8000
WS_ENDPOINT = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/generate"
HTTP_ENDPOINT = f"http://{SERVER_HOST}:{SERVER_PORT}/generate/raw"

OUTPUT_DIR = Path("./websocket_test_output")
TRANSCRIPTS_CACHE_PATH = Path("./audio_data/transcripts_cache.json")

# Will be set during test run
TEST_TEXT = ""
REFERENCE_AUDIO = None
USED_EMOTION = "neutral"


@dataclass
class TestResult:
    """Results from a single test case"""
    test_id: str
    method: str  # "websocket" or "http"
    text: str
    emotion: str
    sentence_count: int
    
    # Timing metrics
    total_time: float
    time_to_first_sentence: Optional[float]  # WebSocket only
    generation_time: float
    
    # Audio metrics
    total_duration: float
    rtf: float
    
    # WebSocket specific
    chunks_received: Optional[int] = None
    average_chunk_time: Optional[float] = None
    
    # File paths
    audio_file: Optional[str] = None


@dataclass 
class WebSocketAudioChunk:
    """Audio chunk received from WebSocket"""
    sentence_index: int
    sentence_text: str
    audio_base64: str
    duration: float
    generation_time: float
    rtf: float
    is_final: bool
    received_at: float


class WebSocketTTSClient:
    """WebSocket client for streaming TTS testing"""
    
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket = None
        
    async def connect(self):
        """Connect to WebSocket server"""
        self.websocket = await websockets.connect(self.uri)
        
    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    async def generate_streaming(
        self, 
        text: str, 
        emotion: str = "neutral",
        temperature: float = 0.8,
        cfg_weight: float = 0.5
    ) -> TestResult:
        """
        Generate streaming TTS and collect all chunks
        """
        if not self.websocket:
            raise RuntimeError("Not connected to WebSocket server")
        
        test_id = str(uuid.uuid4())
        request_id = f"test_{test_id}"
        start_time = time.time()
        
        # Send request
        request = {
            "type": "request",
            "text": text,
            "emotion": emotion,
            "temperature": temperature,
            "cfg_weight": cfg_weight,
            "request_id": request_id,
            "include_progress": True
        }
        
        await self.websocket.send(json.dumps(request))
        
        # Collect responses
        audio_chunks = []
        time_to_first_sentence = None
        sentence_count = 0
        completed = False
        
        try:
            while not completed:
                response_raw = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=60.0  # 60 second timeout
                )
                
                response = json.loads(response_raw)
                response_time = time.time()
                
                if response.get("type") == "audio":
                    # Record time to first sentence
                    if time_to_first_sentence is None:
                        time_to_first_sentence = response_time - start_time
                    
                    chunk = WebSocketAudioChunk(
                        sentence_index=response["sentence_index"],
                        sentence_text=response["sentence_text"],
                        audio_base64=response["audio_base64"],
                        duration=response["duration"],
                        generation_time=response["generation_time"],
                        rtf=response["rtf"],
                        is_final=response.get("is_final", False),
                        received_at=response_time
                    )
                    audio_chunks.append(chunk)
                    
                    if chunk.is_final:
                        completed = True
                
                elif response.get("type") == "complete":
                    sentence_count = response["total_sentences"]
                    completed = True
                    
                elif response.get("type") == "error":
                    raise RuntimeError(f"Server error: {response['error']}")
        
        except asyncio.TimeoutError:
            raise RuntimeError("WebSocket request timed out")
        
        total_time = time.time() - start_time
        
        # Concatenate audio chunks with improved quality
        audio_data = []
        total_duration = 0.0
        total_generation_time = 0.0
        
        for i, chunk in enumerate(audio_chunks):
            # Decode base64 audio
            audio_bytes = base64.b64decode(chunk.audio_base64)
            
            # Load audio tensor
            buffer = io.BytesIO(audio_bytes)
            audio_tensor, sample_rate = torchaudio.load(buffer, format="wav")
            audio_data.append(audio_tensor)
            
            # Add small gap between sentences (except after last)
            if i < len(audio_chunks) - 1:
                gap = torch.zeros(1, int(sample_rate * 0.1))  # 100ms gap
                audio_data.append(gap)
            
            total_duration += chunk.duration
            total_generation_time += chunk.generation_time
        
        # Concatenate all audio
        if audio_data:
            full_audio = torch.cat(audio_data, dim=1)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = OUTPUT_DIR / f"websocket_{test_id}_{emotion}_{timestamp}.wav"
            torchaudio.save(str(audio_file), full_audio, sample_rate)
        else:
            audio_file = None
            full_audio = None
        
        # Calculate metrics
        rtf = total_generation_time / total_duration if total_duration > 0 else 0
        average_chunk_time = sum(c.generation_time for c in audio_chunks) / len(audio_chunks) if audio_chunks else 0
        
        return TestResult(
            test_id=test_id,
            method="websocket",
            text=text,
            emotion=emotion,
            sentence_count=len(audio_chunks),
            total_time=total_time,
            time_to_first_sentence=time_to_first_sentence,
            generation_time=total_generation_time,
            total_duration=total_duration,
            rtf=rtf,
            chunks_received=len(audio_chunks),
            average_chunk_time=average_chunk_time,
            audio_file=str(audio_file) if audio_file else None
        )


class HTTPTTSClient:
    """HTTP client for comparison testing"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def generate_blocking(
        self,
        text: str,
        emotion: str = "neutral", 
        temperature: float = 0.8,
        cfg_weight: float = 0.5
    ) -> TestResult:
        """Generate TTS using blocking HTTP endpoint"""
        
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Send HTTP request
        request_data = {
            "text": text,
            "emotion": emotion,
            "temperature": temperature,
            "cfg_weight": cfg_weight
        }
        
        response = requests.post(self.base_url, json=request_data)
        response.raise_for_status()
        
        total_time = time.time() - start_time
        
        # Parse response headers
        duration = float(response.headers.get("X-Duration", 0))
        generation_time = float(response.headers.get("X-Generation-Time", 0))
        rtf = float(response.headers.get("X-RTF", 0))
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_file = OUTPUT_DIR / f"http_{test_id}_{emotion}_{timestamp}.wav"
        
        with open(audio_file, 'wb') as f:
            f.write(response.content)
        
        # Estimate sentence count (rough)
        estimated_sentences = len([s for s in text.split('.') if s.strip()])
        
        return TestResult(
            test_id=test_id,
            method="http",
            text=text,
            emotion=emotion,
            sentence_count=estimated_sentences,
            total_time=total_time,
            time_to_first_sentence=None,  # N/A for blocking
            generation_time=generation_time,
            total_duration=duration,
            rtf=rtf,
            chunks_received=None,
            average_chunk_time=None,
            audio_file=str(audio_file)
        )


async def check_server_health():
    """Check if the server is running and healthy"""
    try:
        response = requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/health", timeout=5.0)
        response.raise_for_status()
        health_data = response.json()
        
        print(f"✓ Server is healthy")
        print(f"  Model: {health_data['model']}")
        print(f"  Available emotions: {', '.join(health_data['emotions'])}")
        print(f"  Currently processing: {health_data['processing']}")
        
        return health_data['emotions']
        
    except Exception as e:
        print(f"✗ Server health check failed: {e}")
        return None


def clear_output_directory():
    """Clear and recreate the output directory"""
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Cleared output directory: {OUTPUT_DIR}")


def load_transcripts_cache() -> Dict[str, Dict]:
    """Load transcripts from cache file"""
    if not TRANSCRIPTS_CACHE_PATH.exists():
        raise FileNotFoundError(f"Transcripts cache not found: {TRANSCRIPTS_CACHE_PATH}")
    
    with open(TRANSCRIPTS_CACHE_PATH, 'r') as f:
        return json.load(f)


def generate_combined_test_text(transcripts_cache: Dict[str, Dict], count: int = 2) -> tuple:
    """Generate a combined text from random transcripts"""
    available_files = list(transcripts_cache.keys())
    if len(available_files) < count:
        raise ValueError(f"Not enough transcripts available. Need {count}, have {len(available_files)}")
    
    # Select random transcripts
    selected_files = random.sample(available_files, count)
    transcripts = [transcripts_cache[file]["transcript"] for file in selected_files]
    
    # Combine with clear sentence separators
    combined_text = " ".join(transcripts)
    
    print(f"Selected transcripts from: {', '.join([Path(f).stem for f in selected_files])}")
    print(f"Individual transcripts:")
    for i, transcript in enumerate(transcripts, 1):
        print(f"  {i}. {transcript[:80]}{'...' if len(transcript) > 80 else ''}")
    print(f"Combined length: {len(combined_text)} characters")
    
    return combined_text, transcripts


async def run_performance_tests():
    """Run comprehensive performance tests"""
    global TEST_TEXT, REFERENCE_AUDIO, USED_EMOTION
    
    print("="*80)
    print("WEBSOCKET STREAMING TTS PERFORMANCE TEST")
    print("Single Long Audio Generation from 2 Transcriptions")
    print("="*80)
    
    # Clear output directory
    clear_output_directory()
    
    # Load and prepare test data
    print("\nLoading transcripts cache...")
    try:
        transcripts_cache = load_transcripts_cache()
        print(f"Loaded {len(transcripts_cache)} cached transcripts")
        
        # Generate combined test text
        TEST_TEXT, individual_transcripts = generate_combined_test_text(transcripts_cache, 2)
        
    except Exception as e:
        print(f"Error loading transcripts: {e}")
        return None
    
    # Check server health
    available_emotions = await check_server_health()
    if not available_emotions:
        print("Server is not available. Please start the server and try again.")
        return
    
    # Use first available emotion
    USED_EMOTION = available_emotions[0] if available_emotions else "neutral"
    
    print(f"\nUsing emotion: {USED_EMOTION}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Initialize clients
    ws_client = WebSocketTTSClient(WS_ENDPOINT)
    http_client = HTTPTTSClient(HTTP_ENDPOINT)
    
    all_results = []
    
    try:
        # Connect to WebSocket
        print(f"\nConnecting to WebSocket at {WS_ENDPOINT}")
        await ws_client.connect()
        print("✓ WebSocket connected")
        
        print(f"\n--- WebSocket Streaming Test ---")
        print(f"Text length: {len(TEST_TEXT)} characters")
        print(f"Estimated sentences: {len([s for s in TEST_TEXT.split('.') if s.strip()])}")
        
        # WebSocket test
        print("\nGenerating via WebSocket streaming...")
        try:
            ws_result = await ws_client.generate_streaming(TEST_TEXT, USED_EMOTION)
            all_results.append(ws_result)
            
            print(f"  ✓ Time-to-First-Sentence: {ws_result.time_to_first_sentence:.3f}s")
            print(f"  ✓ Total Generation Time: {ws_result.total_time:.3f}s")
            print(f"  ✓ Sentences Processed: {ws_result.chunks_received}")
            print(f"  ✓ Audio Duration: {ws_result.total_duration:.2f}s")
            print(f"  ✓ Real-Time Factor: {ws_result.rtf:.3f}")
            print(f"  ✓ Audio saved to: {ws_result.audio_file}")
            
        except Exception as e:
            print(f"  ✗ WebSocket test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # HTTP test for comparison
        print(f"\n--- HTTP Blocking Test ---")
        print("Generating via HTTP blocking...")
        try:
            http_result = http_client.generate_blocking(TEST_TEXT, USED_EMOTION)
            all_results.append(http_result)
            
            print(f"  ✓ Total Generation Time: {http_result.total_time:.3f}s")
            print(f"  ✓ Audio Duration: {http_result.total_duration:.2f}s")
            print(f"  ✓ Real-Time Factor: {http_result.rtf:.3f}")
            print(f"  ✓ Audio saved to: {http_result.audio_file}")
            
        except Exception as e:
            print(f"  ✗ HTTP test failed: {e}")
            import traceback
            traceback.print_exc()
        
    finally:
        await ws_client.disconnect()
    
    # Analyze and display results
    display_results(all_results)
    
    return all_results


def display_results(results: List[TestResult]):
    """Display comprehensive test results"""
    
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Separate WebSocket and HTTP results
    ws_results = [r for r in results if r.method == "websocket"]
    http_results = [r for r in results if r.method == "http"]
    
    # Summary statistics
    if ws_results:
        avg_ttfs = sum(r.time_to_first_sentence for r in ws_results if r.time_to_first_sentence) / len(ws_results)
        avg_ws_total = sum(r.total_time for r in ws_results) / len(ws_results)
        avg_ws_rtf = sum(r.rtf for r in ws_results) / len(ws_results)
    else:
        avg_ttfs = avg_ws_total = avg_ws_rtf = 0
    
    if http_results:
        avg_http_total = sum(r.total_time for r in http_results) / len(http_results)
        avg_http_rtf = sum(r.rtf for r in http_results) / len(http_results)
    else:
        avg_http_total = avg_http_rtf = 0
    
    print(f"\nSUMMARY STATISTICS:")
    print(f"WebSocket Tests: {len(ws_results)}")
    print(f"HTTP Tests: {len(http_results)}")
    print()
    if ws_results:
        print(f"WebSocket Time-to-First-Sentence: {avg_ttfs:.3f}s")
        print(f"WebSocket Total Time: {avg_ws_total:.3f}s")
        print(f"WebSocket RTF: {avg_ws_rtf:.3f}")
    if http_results:
        print(f"HTTP Total Time: {avg_http_total:.3f}s") 
        print(f"HTTP RTF: {avg_http_rtf:.3f}")
    
    if avg_http_total > 0 and avg_ttfs > 0:
        latency_improvement = ((avg_http_total - avg_ttfs) / avg_http_total) * 100
        print(f"Perceived Latency Improvement: {latency_improvement:.1f}%")
        print(f"(Time to hear first audio: {avg_ttfs:.3f}s vs waiting for full HTTP: {avg_http_total:.3f}s)")
    
    # Detailed results table
    print(f"\nDETAILED RESULTS:")
    
    table_data = []
    for result in results:
        table_data.append([
            result.method.upper(),
            result.emotion,
            f"{len(result.text)}",
            f"{result.sentence_count}",
            f"{result.time_to_first_sentence:.3f}s" if result.time_to_first_sentence else "N/A",
            f"{result.total_time:.3f}s",
            f"{result.total_duration:.2f}s",
            f"{result.rtf:.3f}",
            f"{result.chunks_received}" if result.chunks_received else "1"
        ])
    
    headers = [
        "Method", "Emotion", "Text Len", "Sentences", 
        "TTFS", "Total Time", "Audio Dur", "RTF", "Chunks"
    ]
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # File outputs
    print(f"\nOUTPUT FILES:")
    for result in results:
        if result.audio_file:
            file_size = Path(result.audio_file).stat().st_size if Path(result.audio_file).exists() else 0
            print(f"  {result.method.upper()}: {result.audio_file} ({file_size:,} bytes)")
    
    # Save results to JSON
    results_file = OUTPUT_DIR / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump([
            {
                "test_id": r.test_id,
                "method": r.method,
                "text": r.text,
                "emotion": r.emotion,
                "sentence_count": r.sentence_count,
                "total_time": r.total_time,
                "time_to_first_sentence": r.time_to_first_sentence,
                "generation_time": r.generation_time,
                "total_duration": r.total_duration,
                "rtf": r.rtf,
                "chunks_received": r.chunks_received,
                "average_chunk_time": r.average_chunk_time,
                "audio_file": r.audio_file
            }
            for r in results
        ], f, indent=2)
    
    print(f"  Results JSON: {results_file}")


async def main():
    """Main entry point"""
    try:
        results = await run_performance_tests()
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
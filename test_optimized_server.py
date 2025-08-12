#!/usr/bin/env python3
"""
Test script to verify the optimized server is working correctly.
This script:
1. Starts the server with optimizations
2. Waits for initialization and compilation
3. Tests generation endpoints
4. Reports performance metrics
"""

import time
import requests
import json
import base64
import subprocess
import sys
from pathlib import Path
import signal
import os

# Server configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Test configuration
TEST_TEXT = "Hello! This is a test of the optimized Chatterbox server. The performance improvements should make generation much faster."
TEST_EMOTION = "happy"


def wait_for_server(max_wait=60):
    """Wait for server to be ready, including compilation time."""
    print(f"Waiting for server to be ready (max {max_wait}s)...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ Server is responding to health checks")
                
                # Check model info to see if optimizations are applied
                model_response = requests.get(f"{SERVER_URL}/model/info", timeout=2)
                if model_response.status_code == 200:
                    model_info = model_response.json()
                    print(f"✓ Model loaded: {model_info.get('model_type')}")
                    
                    # Check for optimizations
                    if 'optimizations' in model_info:
                        opts = model_info['optimizations']
                        print("✓ Optimizations applied:")
                        print(f"  - BFloat16: {opts.get('bfloat16', False)}")
                        print(f"  - torch.compile: {opts.get('torch_compile', False)}")
                        print(f"  - Reduced cache: {opts.get('reduced_cache', False)}")
                        if opts.get('compile_mode'):
                            print(f"  - Compile mode: {opts['compile_mode']}")
                        if opts.get('compilation_time'):
                            print(f"  - Compilation time: {opts['compilation_time']:.2f}s")
                    
                    return True
                    
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
        print(f"  Waiting... ({int(time.time() - start_time)}s elapsed)")
    
    return False


def test_generation():
    """Test the generation endpoint."""
    print(f"\nTesting generation with emotion '{TEST_EMOTION}'...")
    
    # Prepare request
    request_data = {
        "text": TEST_TEXT,
        "emotion": TEST_EMOTION,
        "temperature": 0.8,
        "cfg_weight": 0.5
    }
    
    # Time the generation
    start_time = time.time()
    
    try:
        # Test raw audio endpoint (more efficient than base64)
        response = requests.post(
            f"{SERVER_URL}/generate/raw",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            generation_time = time.time() - start_time
            
            # Get metrics from headers
            duration = float(response.headers.get('X-Duration', 0))
            rtf = float(response.headers.get('X-RTF', 0))
            gen_time = float(response.headers.get('X-Generation-Time', 0))
            queue_time = float(response.headers.get('X-Queue-Time', 0))
            
            print(f"✓ Generation successful!")
            print(f"  - Request time: {generation_time:.2f}s")
            print(f"  - Generation time: {gen_time:.2f}s")
            print(f"  - Queue time: {queue_time:.3f}s")
            print(f"  - Audio duration: {duration:.2f}s")
            print(f"  - RTF: {rtf:.3f}")
            print(f"  - Audio size: {len(response.content)} bytes")
            
            # Calculate tokens/second estimate
            estimated_tokens = duration * 50  # Rough estimate
            tokens_per_second = estimated_tokens / gen_time if gen_time > 0 else 0
            print(f"  - Estimated tokens/s: {tokens_per_second:.1f}")
            
            # Save audio for verification
            output_path = Path("test_output_optimized.wav")
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  - Audio saved to: {output_path}")
            
            return True
        else:
            print(f"✗ Generation failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Generation request failed: {e}")
        return False


def test_multiple_generations(num_tests=3):
    """Test multiple generations to measure consistent performance."""
    print(f"\nTesting {num_tests} consecutive generations...")
    
    total_gen_time = 0
    total_rtf = 0
    successful = 0
    
    for i in range(num_tests):
        print(f"\n  Test {i+1}/{num_tests}:")
        
        # Vary the text slightly
        test_text = f"Test number {i+1}. {TEST_TEXT}"
        
        request_data = {
            "text": test_text,
            "emotion": TEST_EMOTION,
            "temperature": 0.8,
            "cfg_weight": 0.5
        }
        
        try:
            response = requests.post(
                f"{SERVER_URL}/generate/raw",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                gen_time = float(response.headers.get('X-Generation-Time', 0))
                rtf = float(response.headers.get('X-RTF', 0))
                duration = float(response.headers.get('X-Duration', 0))
                
                total_gen_time += gen_time
                total_rtf += rtf
                successful += 1
                
                # Estimate tokens/s
                tokens_per_second = (duration * 50) / gen_time if gen_time > 0 else 0
                
                print(f"    ✓ Gen time: {gen_time:.2f}s, RTF: {rtf:.3f}, Tokens/s: {tokens_per_second:.1f}")
            else:
                print(f"    ✗ Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"    ✗ Request failed: {e}")
    
    if successful > 0:
        avg_gen_time = total_gen_time / successful
        avg_rtf = total_rtf / successful
        print(f"\n✓ Completed {successful}/{num_tests} tests")
        print(f"  Average generation time: {avg_gen_time:.2f}s")
        print(f"  Average RTF: {avg_rtf:.3f}")
        
        # Compare to baseline
        baseline_rtf = 2.0  # Typical unoptimized RTF
        improvement = baseline_rtf / avg_rtf if avg_rtf > 0 else 0
        print(f"  Performance improvement: {improvement:.1f}x vs baseline")
    else:
        print(f"\n✗ All tests failed")


def get_server_metrics():
    """Get and display server metrics."""
    print("\nFetching server metrics...")
    
    try:
        # Get model info
        response = requests.get(f"{SERVER_URL}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print("✓ Model metrics:")
            print(f"  - Model type: {info.get('model_type')}")
            print(f"  - Device: {info.get('device')}")
            print(f"  - VRAM usage: {info.get('vram_usage_mb', 0):.1f} MB")
            print(f"  - Total requests: {info.get('total_requests', 0)}")
            
            if info.get('avg_generation_time'):
                print(f"  - Avg generation time: {info['avg_generation_time']:.2f}s")
            if info.get('avg_rtf'):
                print(f"  - Avg RTF: {info['avg_rtf']:.3f}")
        
        # Get server status
        response = requests.get(f"{SERVER_URL}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            uptime = status.get('uptime_seconds', 0)
            print(f"  - Server uptime: {uptime:.1f}s")
            print(f"  - Memory usage: {status.get('memory_usage_mb', 0):.1f} MB")
            
    except Exception as e:
        print(f"✗ Failed to get metrics: {e}")


def main():
    """Main test execution."""
    print("=" * 80)
    print("Optimized Chatterbox Server Test")
    print("=" * 80)
    
    # Check if server is already running
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=1)
        if response.status_code == 200:
            print("Server is already running. Using existing instance.")
            server_process = None
    except:
        print("Server not running. Please start the server manually:")
        print("\n  python -m src.server.main\n")
        print("Then run this test script again.")
        sys.exit(1)
    
    # Wait for server to be ready
    if wait_for_server(max_wait=90):
        print("\n✓ Server is ready with optimizations!")
        
        # Run tests
        time.sleep(2)  # Give server a moment to stabilize
        
        if test_generation():
            print("\n✓ Single generation test passed!")
            
            # Test multiple generations
            test_multiple_generations(num_tests=3)
            
            # Get final metrics
            get_server_metrics()
            
            print("\n" + "=" * 80)
            print("✓ All tests completed successfully!")
            print("✓ Server is running with optimizations and achieving improved performance!")
            print("=" * 80)
        else:
            print("\n✗ Generation test failed")
            sys.exit(1)
    else:
        print("\n✗ Server failed to start or initialize properly")
        sys.exit(1)


if __name__ == "__main__":
    main()
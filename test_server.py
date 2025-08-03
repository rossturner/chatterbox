#!/usr/bin/env python3
"""
Test script for the Chatterbox TTS API server.
"""

import requests
import time
import json
from pathlib import Path

def test_server():
    """Test the server functionality."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Chatterbox TTS API Server")
    print("=" * 50)
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Server is healthy")
            print(f"   📊 Status: {health['status']}")
            print(f"   🧠 Model loaded: {health['model_loaded']}")
            print(f"   😊 Emotions loaded: {health['emotions_loaded']}")
            print(f"   💾 Memory usage: {health['memory_usage_mb']:.1f}MB")
            print(f"   🖥️  Device: {health['device']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test emotions list
    print("\n2. Testing emotions endpoint...")
    try:
        response = requests.get(f"{base_url}/emotions")
        if response.status_code == 200:
            emotions_data = response.json()
            emotions = emotions_data['emotions']
            print(f"   ✅ Found {len(emotions)} emotion profiles")
            
            for emotion in emotions:
                print(f"   😊 {emotion['character']} - {emotion['name']} (exaggeration: {emotion['exaggeration']})")
                print(f"      Voice samples: {len(emotion['voice_samples'])}")
            
            if not emotions:
                print("   ⚠️  No emotions found - server will have limited functionality")
                return True
                
        else:
            print(f"   ❌ Emotions list failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Emotions list error: {e}")
        return False
    
    # Test generation with first available emotion
    if emotions:
        first_emotion = emotions[0]
        print(f"\n3. Testing TTS generation with '{first_emotion['name']}'...")
        
        try:
            generation_request = {
                "text": "Hello! This is a test of the Chatterbox TTS API server. The voice cloning and emotion system appears to be working correctly.",
                "emotion": first_emotion['id'],
                "temperature": 0.8,
                "cfg_weight": 0.5
            }
            
            print("   🎵 Generating speech...")
            start_time = time.time()
            
            response = requests.post(f"{base_url}/generate", json=generation_request)
            
            if response.status_code == 200:
                result = response.json()
                generation_time = time.time() - start_time
                
                print(f"   ✅ Generation successful!")
                print(f"   ⏱️  Total time: {generation_time:.2f}s")
                print(f"   🎼 Audio duration: {result['duration_seconds']:.2f}s")
                print(f"   ⚡ Generation time: {result['generation_time_seconds']:.2f}s")
                print(f"   📈 Real-time factor: {result['metadata']['rtf']:.2f}x")
                print(f"   📁 Audio URL: {result['audio_url']}")
                
                # Test file access
                if result['audio_url']:
                    file_response = requests.head(base_url + result['audio_url'])
                    if file_response.status_code == 200:
                        print(f"   ✅ Audio file accessible")
                    else:
                        print(f"   ⚠️  Audio file not accessible: {file_response.status_code}")
                        
            else:
                error_data = response.json()
                print(f"   ❌ Generation failed: {error_data.get('detail', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            return False
    
    # Test emotion creation
    print("\n4. Testing emotion creation...")
    try:
        new_emotion = {
            "name": "Test Emotion",
            "character": "TestBot",
            "exaggeration": 0.5,
            "description": "A test emotion profile"
        }
        
        response = requests.post(f"{base_url}/emotions", json=new_emotion)
        
        if response.status_code == 200:
            created_emotion = response.json()
            print(f"   ✅ Created emotion: {created_emotion['id']}")
            
            # Clean up - delete the test emotion
            delete_response = requests.delete(f"{base_url}/emotions/{created_emotion['id']}")
            if delete_response.status_code == 200:
                print(f"   🗑️  Cleaned up test emotion")
            else:
                print(f"   ⚠️  Could not clean up test emotion")
                
        else:
            error_data = response.json()
            print(f"   ❌ Emotion creation failed: {error_data.get('detail', 'Unknown error')}")
            
    except Exception as e:
        print(f"   ❌ Emotion creation error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Server test completed successfully!")
    print(f"🌐 Web interface: {base_url}")
    print(f"📚 API docs: {base_url}/docs")
    
    return True


def main():
    """Main test function."""
    try:
        success = test_server()
        if success:
            print("\n🎉 All tests passed! Server is working correctly.")
        else:
            print("\n❌ Some tests failed. Check server logs for details.")
            exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test script error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
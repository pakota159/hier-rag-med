#!/usr/bin/env python3
"""
Test Ollama API endpoints to find the correct one.
"""

import requests
import json

def test_ollama_endpoints():
    """Test different Ollama API endpoints."""
    print("🔍 Testing Ollama API endpoints...")
    
    base_url = "http://localhost:11434"
    
    # Test 1: Check if Ollama is running
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running. Available models: {[m['name'] for m in models]}")
        else:
            print(f"❌ Ollama tags endpoint returned: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return
    
    # Test 2: Try the generate endpoint (older API)
    print("\n🧪 Testing /api/generate endpoint...")
    generate_payload = {
        "model": "mistral:7b-instruct",
        "prompt": "Hello, how are you?",
        "stream": False
    }
    
    try:
        response = requests.post(f"{base_url}/api/generate", json=generate_payload, timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Generate API works! Response: {result.get('response', '')[:100]}...")
        else:
            print(f"   ❌ Generate API failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Generate API error: {e}")
    
    # Test 3: Try the chat endpoint (newer API)
    print("\n🧪 Testing /api/chat endpoint...")
    chat_payload = {
        "model": "mistral:7b-instruct",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "stream": False
    }
    
    try:
        response = requests.post(f"{base_url}/api/chat", json=chat_payload, timeout=30)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", {}).get("content", "")
            print(f"   ✅ Chat API works! Response: {message[:100]}...")
        else:
            print(f"   ❌ Chat API failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Chat API error: {e}")
    
    # Test 4: Check Ollama version
    print("\n🔍 Checking Ollama version...")
    try:
        response = requests.get(f"{base_url}/api/version", timeout=5)
        if response.status_code == 200:
            version = response.json()
            print(f"   Ollama version: {version}")
        else:
            print(f"   Version check failed: {response.status_code}")
    except Exception as e:
        print(f"   Version check error: {e}")

if __name__ == "__main__":
    test_ollama_endpoints()
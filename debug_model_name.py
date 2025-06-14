#!/usr/bin/env python3
"""Debug what model name is actually being used"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import Config
from generation import Generator

def debug_model_config():
    """Debug the model configuration."""
    print("🔍 Debugging model configuration...\n")
    
    # Load config
    config = Config(Path("config.yaml"))
    
    print("1. Raw config file contents:")
    with open("config.yaml", "r") as f:
        content = f.read()
    print(content)
    
    print("\n2. Parsed config:")
    print(f"   Full config: {config.config}")
    
    print("\n3. Models section:")
    models = config.config.get("models", {})
    print(f"   Models: {models}")
    
    print("\n4. LLM section:")
    llm_config = models.get("llm", {})
    print(f"   LLM config: {llm_config}")
    
    print("\n5. Model name:")
    model_name = llm_config.get("name", "NOT FOUND")
    print(f"   Model name: '{model_name}'")
    print(f"   Model name type: {type(model_name)}")
    print(f"   Model name repr: {repr(model_name)}")
    
    print("\n6. Generator initialization:")
    try:
        generator = Generator(config)
        print(f"   Generator model name: '{generator.model_name}'")
        print(f"   Generator model name type: {type(generator.model_name)}")
        print(f"   Generator model name repr: {repr(generator.model_name)}")
        
        # Test API call payload
        test_payload = {
            "model": generator.model_name,
            "prompt": "test",
            "stream": False
        }
        print(f"\n7. API payload:")
        print(f"   Payload: {test_payload}")
        
    except Exception as e:
        print(f"   ❌ Generator initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_config()
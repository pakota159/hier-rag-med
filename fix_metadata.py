#!/usr/bin/env python3
"""
Fix metadata in existing KG dataset for ChromaDB compatibility.
"""

import json
from pathlib import Path

def fix_metadata(data_file: Path):
    """Fix list metadata to strings."""
    print(f"🔧 Fixing metadata in {data_file}...")
    
    with open(data_file, "r") as f:
        documents = json.load(f)
    
    fixed_count = 0
    for doc in documents:
        if "metadata" in doc:
            metadata = doc["metadata"]
            for key, value in metadata.items():
                if isinstance(value, list):
                    # Convert lists to comma-separated strings
                    metadata[key] = ", ".join(str(item) for item in value) if value else ""
                    fixed_count += 1
    
    # Save fixed data
    with open(data_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    print(f"✅ Fixed {fixed_count} list metadata fields")
    print(f"📄 Updated {len(documents)} documents")

if __name__ == "__main__":
    # Fix the combined dataset
    combined_file = Path("data/kg_raw/combined/all_medical_data.json")
    if combined_file.exists():
        fix_metadata(combined_file)
        print("🎉 Ready to run KG Streamlit app!")
    else:
        print("❌ Combined dataset not found")
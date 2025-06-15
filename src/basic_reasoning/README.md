# Basic Reasoning Module

A comprehensive module for collecting and processing medical reasoning datasets for the HierRAGMed system.

## Overview

This module provides tools for collecting, processing, and validating medical reasoning datasets from various sources:

- **MedReason**: Knowledge graph-guided reasoning chains
- **MSDiagnosis**: Multi-step diagnostic scenarios
- **PMC Patients**: Patient case studies from medical literature
- **DrugBank**: Comprehensive drug information

## Features

- Unified document format for all datasets
- Rate-limited API access
- Checkpoint-based data collection
- Comprehensive validation
- Progress tracking
- Detailed logging
- Sample data generation for testing

## Installation

1. Ensure you have Python 3.10+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The module is configured through `config/datasets_config.yaml`. Key configurations include:

- Dataset sources and URLs
- API endpoints and rate limits
- Expected dataset sizes
- Processing parameters
- Output directory structure

## Usage

### Basic Usage

```python
from basic_reasoning.run_data_collection import run_data_collection
import yaml

# Load configuration
with open("config/datasets_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Run data collection
run_data_collection(config)
```

### Command Line Interface

```bash
# Run with default configuration
python -m basic_reasoning.run_data_collection

# Run with custom configuration
python -m basic_reasoning.run_data_collection --config path/to/config.yaml

# Limit number of documents
python -m basic_reasoning.run_data_collection --max-documents 1000
```

## Output Structure

```
data/
├── foundation_dataset/
│   ├── checkpoints/
│   │   ├── medreason_*.json
│   │   ├── msdiagnosis_*.json
│   │   ├── pmc_patients_*.json
│   │   └── drugbank_*.json
│   ├── medreason_documents.json
│   ├── msdiagnosis_documents.json
│   ├── pmc_patients_documents.json
│   ├── drugbank_documents.json
│   ├── unified_dataset.json
│   └── validation_report.txt
└── logs/
    ├── data_collection_*.log
    ├── medreason_*.log
    ├── msdiagnosis_*.log
    ├── pmc_patients_*.log
    ├── drugbank_*.log
    └── validation_*.log
```

## Document Format

All documents follow a unified format:

```python
{
    "text": str,  # Main document text
    "metadata": {
        "title": str,
        "authors": List[str],
        "publication_date": str,
        "medical_specialty": str,
        "evidence_level": str,
        "reasoning_chain": List[Dict]  # Optional reasoning steps
    },
    "source_dataset": str,  # Dataset identifier
    "doc_id": str,  # Unique document identifier
    "reasoning_type": str  # Type of reasoning (e.g., "diagnostic", "case_study")
}
```

## Validation

The module includes comprehensive validation:

- Required field checking
- Text length validation
- Metadata completeness
- Reasoning chain structure
- Dataset-specific validation rules

Validation reports include:
- Overall statistics
- Validation error counts
- Reasoning type distribution
- Medical specialty distribution

## Development

### Adding New Datasets

1. Create a new fetcher class inheriting from `BaseFetcher`
2. Implement required methods:
   - `get_dataset_info()`
   - `fetch_documents()`
3. Add dataset configuration to `datasets_config.yaml`
4. Register fetcher in `run_data_collection.py`

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=basic_reasoning tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
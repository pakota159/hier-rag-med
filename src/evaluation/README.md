```
cd /workspace
git clone https://github.com/pakota159/hier-rag-med.git hierragmed
```

```
bash src/evaluation/runpod_setup.sh
```

```
source /opt/miniconda/etc/profile.d/conda.sh
```

```
conda activate hierragmed-gpu
```

# 1. Fetch KG enhanced data for better performance
python fetch_data.py --source all --max-results 1000

# 2. Fetch foundation data (required for hierarchical system)
# Best performance: 70-75% MIRAGE expected
python fetch_foundation_data.py --max-results 5000

# 3. Setup KG system collections
python setup_kg_system.py

# 4. Setup Hierarchical system collections
python setup_hierarchical_system.py

# Quick test with mock systems
python src/evaluation/run_evaluation.py --quick

# Test individual systems
python src/evaluation/run_evaluation.py --quick --models kg_system
python src/evaluation/run_evaluation.py --quick --models hierarchical_system

# Full test
python src/evaluation/run_evaluation.py --models hierarchical_system

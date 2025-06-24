```
cd /workspace
git clone https://github.com/pakota159/hier-rag-med.git hierragmed
git submodule update --init --recursive
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
python fetch_foundation_data.py --max-results 50000

# 3. Setup KG system collections
python setup_kg_system.py

# 4. Setup Hierarchical system collections
python setup_hierarchical_system.py

# Quick evaluation of hierarchical system on MIRAGE only
python src/evaluation/run_evaluation.py --quick --models hierarchical_system --benchmark mirage

# Full evaluation of hierarchical system on MIRAGE only
python src/evaluation/run_evaluation.py --full --models hierarchical_system --benchmark mirage

# Quick evaluation on all benchmarks
python src/evaluation/run_evaluation.py --quick --models hierarchical_system

# Full evaluation on all benchmarks
python src/evaluation/run_evaluation.py --full --models hierarchical_system

# Standard mode (config defaults)
python src/evaluation/run_evaluation.py --models hierarchical_system

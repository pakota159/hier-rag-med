```
bash src/evaluation/runpod_setup.sh
```

```
source /opt/miniconda/etc/profile.d/conda.sh
```

```
conda activate hierragmed-gpu
```

# Quick test with mock systems
python src/evaluation/run_evaluation.py --quick

# Test specific models
python src/evaluation/run_evaluation.py --models kg_system

# Full test
python src/evaluation/run_evaluation.py --full

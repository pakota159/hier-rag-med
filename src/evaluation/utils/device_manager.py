# Quick Fixes for HierRAGMed Evaluation Issues

# Fix 1: GPU Device Manager setup_gpu_optimization method
# In src/evaluation/utils/device_manager.py

def setup_gpu_optimization(self) -> None:
    """Setup GPU optimizations for evaluation."""
    if not self.is_available:
        return
    
    import os
    import torch
    
    # Set CUDA environment variables for RTX 4090 optimization
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": str(self.device_id),
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",  # Async CUDA operations
        "TOKENIZERS_PARALLELISM": "false",  # Avoid tokenizer warnings
    })
    
    # Enable CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul on Ampere
        torch.backends.cudnn.allow_tf32 = True        # Faster convolutions on Ampere
        torch.backends.cudnn.benchmark = True         # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False    # Faster but non-deterministic
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    logger.info("🔧 GPU optimizations configured")


# Fix 2: JSON Serialization for GPUInfo
# In updated run_evaluation.py, modify the save results section:

# Replace this section in run_evaluation function:
def save_evaluation_results(all_results, results_dir, models_filter):
    """Save evaluation results with proper JSON serialization."""
    
    # Convert GPUInfo to serializable dict
    if "environment" in all_results and "gpu_info" in all_results["environment"]:
        gpu_info = all_results["environment"]["gpu_info"]
        all_results["environment"]["gpu_info"] = {
            "name": gpu_info.name,
            "memory_total": gpu_info.memory_total,
            "memory_free": gpu_info.memory_free,
            "memory_used": gpu_info.memory_used,
            "cuda_version": gpu_info.cuda_version,
            "device_id": gpu_info.device_id
        }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if models_filter:
        model_suffix = "_".join(models_filter)
        results_file = results_dir / f"evaluation_results_{model_suffix}_{timestamp}.json"
    else:
        results_file = results_dir / f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return results_file


# Fix 3: MS MARCO evaluation error fix
# In src/evaluation/benchmarks/msmarco_benchmark.py, fix the evaluate_response method:

def evaluate_response(self, question: Dict, response: str, retrieved_docs: List[Dict]) -> Dict:
    """Evaluate a model response for MS MARCO benchmark."""
    logger.debug(f"🔍 Evaluating MS MARCO retrieval for query {question.get('question_id', 'unknown')}")
    
    metrics = {}
    
    try:
        # Fix: Ensure response is a string
        if isinstance(response, dict):
            response = str(response)
        elif response is None:
            response = ""
        
        # Fix: Ensure query is a string
        query = question.get("query", question.get("question", ""))
        if isinstance(query, dict):
            query = str(query)
        
        # Continue with evaluation...
        # ... rest of the method
        
    except Exception as e:
        logger.error(f"❌ MS MARCO evaluation failed: {e}")
        return {
            "mrr": 0.0,
            "ndcg": 0.0,
            "map": 0.0,
            "retrieval_accuracy": 0.0,
            "passage_quality": 0.0,
            "relevance_score": 0.0,
            "overall_score": 0.0,
            "error": str(e)
        }


# Fix 4: Download MIRAGE benchmark.json
# Run this command to fix MIRAGE data loading:

"""
cd mirage
wget https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json
cd ..
"""

# Fix 5: Accuracy calculation fix in base_benchmark.py
# The issue is in calculate_metrics method:

def calculate_metrics(self, results: List[Dict]) -> Dict:
    """Calculate aggregate metrics from individual results."""
    if not results:
        return {"error": "No results to calculate metrics"}
    
    # Add sample information to metrics
    sample_info = self.get_sample_info()
    
    # Calculate basic metrics - FIX: Look for overall_score field
    total_score = 0
    valid_results = 0
    correct_count = 0
    
    for r in results:
        if "error" not in r and r.get("status") != "failed":
            # Use overall_score if available, otherwise use score
            score = r.get("overall_score", r.get("score", 0))
            total_score += score
            valid_results += 1
            
            # Consider correct if score > 0.5 or if explicitly marked correct
            if score > 0.5 or r.get("correct", False):
                correct_count += 1
    
    # Calculate averages
    avg_score = total_score / valid_results if valid_results > 0 else 0
    accuracy = (correct_count / valid_results * 100) if valid_results > 0 else 0
    
    metrics = {
        "total_questions": len(results),
        "valid_results": valid_results,
        "average_score": avg_score,
        "accuracy": accuracy,  # This should now show proper percentage
        "correct_answers": correct_count,
        "benchmark_name": self.name,
        "sample_info": sample_info
    }
    
    # Log sample utilization
    utilization = sample_info["utilization_percentage"]
    if utilization < 100:
        logger.info(f"   📊 {self.name}: Used {sample_info['effective_sample_size']} of {sample_info['total_available']} questions ({utilization:.1f}%)")
    else:
        logger.info(f"   📊 {self.name}: Used FULL dataset ({sample_info['total_available']} questions)")
    
    return metrics


# Commands to implement fixes:

"""
# 1. Download MIRAGE data
cd mirage
wget https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json

# 2. Apply the DeviceManager fix to src/evaluation/utils/device_manager.py
# 3. Apply the JSON serialization fix to run_evaluation.py  
# 4. Apply the MS MARCO fix to src/evaluation/benchmarks/msmarco_benchmark.py
# 5. Apply the accuracy calculation fix to src/evaluation/benchmarks/base_benchmark.py

# Then re-run:
python src/evaluation/run_evaluation.py --quick --models hierarchical_system
"""
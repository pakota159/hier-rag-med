# HierRAGMed GPU Evaluation on RunPod

Complete step-by-step guide to deploy and run HierRAGMed evaluation on RunPod GPU instances.

## 🚀 Prerequisites

- RunPod account with GPU instance access
- RTX 4090, RTX 3080/3090, A100, or V100 GPU recommended
- At least 24GB GPU memory for optimal performance
- SSH client (Terminal, PuTTY, etc.)

## 📋 Step-by-Step Deployment Guide

### Step 1: Launch RunPod GPU Instance

1. **Log into RunPod Console**
   - Go to [RunPod.io](https://runpod.io) and sign in
   - Navigate to "Pods" section

2. **Create New Pod**
   - Click "Deploy" or "Create Pod"
   - Select a GPU template (RTX 4090 recommended)
   - Choose disk size: **50GB minimum** (100GB recommended)
   - Enable **SSH access**
   - Note the SSH connection details

3. **Get Connection Information**
   - Copy the SSH command provided (format: `ssh root@<ip> -p <port>`)
   - Save the pod IP address for later Streamlit access

### Step 2: Connect via SSH

```bash
# Connect to your RunPod instance
ssh root@<pod-ip> -p <port>
```

Our global hostname: `ag851js8ghvjgd.runpod.internal`

**What this does:** Establishes secure shell connection to your RunPod GPU instance with root privileges.

### Step 3: Verify GPU Availability

```bash
# Check GPU status
nvidia-smi
```

**What this does:** Verifies NVIDIA drivers are installed and GPU is accessible. You should see your GPU model (e.g., RTX 4090) and memory information.

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  RTX 4090           Off  | 00000000:01:00.0 Off |                  Off |
| 30%   45C    P8    15W / 450W |      0MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Step 4: Upload HierRAGMed Source Code

```bash
# Option A: Clone from Git (recommended)
cd /workspace
git clone <your-hierragmed-repository-url> hierragmed

# Option B: Upload via SCP (from your local machine)
# scp -P <port> -r /path/to/local/hierragmed root@<pod-ip>:/workspace/hierragmed
```

**What this does:** Downloads or uploads your HierRAGMed source code to the RunPod instance workspace directory.

### Step 5: Run Environment Setup

```bash
# Navigate to project directory
cd /workspace/hierragmed

# Make setup script executable
chmod +x src/evaluation/runpod_setup.sh

# Run the complete setup
bash src/evaluation/runpod_setup.sh
```

**What this does:** 
- Installs Miniconda Python environment manager
- Creates `hierragmed-gpu` conda environment with Python 3.10
- Installs PyTorch with CUDA 11.8 support
- Installs all required Python packages (transformers, streamlit, etc.)
- Sets up Ollama for local LLM inference
- Creates GPU-optimized configuration files
- Generates startup and monitoring scripts
- Verifies complete installation

This process takes **10-15 minutes** and includes comprehensive verification.

### Step 6: Verify Installation

The setup script automatically runs verification, but you can manually check:

```bash
# Activate the conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hierragmed-gpu

# Verify GPU access from Python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**What this does:** Confirms PyTorch can access the GPU and all packages are properly installed.

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090
```

### Step 7: Start GPU Monitoring (Optional but Recommended)

Open a **second SSH session** to monitor GPU usage:

```bash
# Connect second SSH session
ssh root@<pod-ip> -p <port>

# Run GPU monitoring
/workspace/hierragmed/monitor_gpu.sh
```

**What this does:** Opens a real-time GPU monitoring dashboard showing memory usage, temperature, and utilization. Updates every 2 seconds.

### Step 8: Start HierRAGMed Evaluation

In your **first SSH session**:

```bash
# Start the evaluation system
/workspace/hierragmed/start_evaluation.sh
```

**What this does:**
- Activates the conda environment
- Sets optimal CUDA environment variables
- Starts Ollama LLM service in background
- Launches Streamlit web interface on port 8501
- Configures GPU-optimized settings for RTX 4090

You should see output like:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://0.0.0.0:8501
```

### Step 9: Access Streamlit Interface

1. **Find your RunPod instance IP**
   - Check RunPod console for your pod's IP address
   - Or run: `curl ifconfig.me` in SSH session

2. **Open web browser and navigate to:**
   ```
   http://<your-runpod-ip>:8501
   ```

**What this does:** Opens the HierRAGMed evaluation web interface where you can configure and run evaluations through a user-friendly GUI.

### Step 10: Configure and Run Evaluation

In the Streamlit interface:

1. **Select Benchmarks:**
   - ✅ MIRAGE (Clinical + Research QA)
   - ✅ MedReason (Clinical Reasoning) 
   - ✅ PubMedQA (Research Literature QA)
   - ✅ MS MARCO (Passage Retrieval)

2. **Select Models:**
   - ✅ KG System (Knowledge Graph enhanced)
   - ✅ Hierarchical System (Hierarchical RAG)

3. **Configure GPU Settings:**
   - **Embedding Batch Size:** 128 (RTX 4090 optimized)
   - **LLM Batch Size:** 32 (RTX 4090 optimized)
   - ✅ **Mixed Precision:** Enabled for faster processing
   - ✅ **Enable Checkpointing:** Save progress during evaluation

4. **Start Evaluation:**
   - Click "🚀 Start GPU Evaluation"
   - Monitor progress in real-time
   - View GPU metrics and performance stats

### Step 11: Monitor Evaluation Progress

**In Streamlit Interface:**
- Real-time progress bars
- GPU memory usage metrics
- Temperature and utilization graphs
- Time estimates and throughput stats

**In SSH Monitor Session:**
- Live nvidia-smi output
- Memory allocation tracking
- Temperature monitoring

**In Main SSH Session:**
- Detailed logs and console output
- Error messages and warnings
- Checkpoint save confirmations

## 🛠️ Alternative Command-Line Evaluation

If you prefer command-line instead of Streamlit:

```bash
# Activate environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hierragmed-gpu

# Run evaluation directly
cd /workspace/hierragmed
python src/evaluation/run_evaluation.py
```

## 📊 Accessing Results

Results are automatically saved to:
- **Results Directory:** `/workspace/hierragmed/evaluation/results/`
- **Logs Directory:** `/workspace/hierragmed/evaluation/logs/`
- **Reports:** Generated HTML and JSON files

```bash
# View results
ls -la /workspace/hierragmed/evaluation/results/

# Download results to local machine (from local terminal)
scp -P <port> -r root@<pod-ip>:/workspace/hierragmed/evaluation/results/ ./hierragmed_results/
```

## 🔧 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Restart if needed
sudo systemctl restart nvidia-persistenced
```

### Out of Memory Errors
```bash
# Reduce batch sizes in Streamlit interface
# Or edit config file
nano /workspace/hierragmed/src/evaluation/configs/gpu_runpod_config.yaml
```

### Port 8501 Not Accessible
```bash
# Check if Streamlit is running
ps aux | grep streamlit

# Restart Streamlit
pkill -f streamlit
/workspace/hierragmed/start_evaluation.sh
```

### Conda Environment Issues
```bash
# Recreate environment
conda env remove -n hierragmed-gpu
bash /workspace/hierragmed/src/evaluation/runpod_setup.sh
```

## ⚡ Performance Optimization

### RTX 4090 Optimal Settings:
- **Embedding Batch Size:** 128
- **LLM Batch Size:** 32  
- **Mixed Precision:** Enabled
- **Model Compilation:** Enabled
- **Memory Fraction:** 85% (20GB of 24GB)

### RTX 3080/3090 Settings:
- **Embedding Batch Size:** 64
- **LLM Batch Size:** 16
- **Mixed Precision:** Enabled
- **Memory Fraction:** 80%

## 📋 Quick Command Reference

```bash
# Essential commands
nvidia-smi                                    # Check GPU status
/workspace/hierragmed/monitor_gpu.sh         # GPU monitoring
/workspace/hierragmed/start_evaluation.sh    # Start evaluation
conda activate hierragmed-gpu                # Activate environment
htop                                         # System monitoring
df -h                                        # Disk usage
```

## 🎯 Expected Performance

**RTX 4090 Performance Estimates:**
- **MIRAGE Benchmark:** ~15-20 minutes
- **MedReason Benchmark:** ~10-15 minutes  
- **PubMedQA Benchmark:** ~20-30 minutes
- **MS MARCO Benchmark:** ~30-45 minutes
- **Total Evaluation Time:** ~1.5-2 hours

**Memory Usage:**
- **Peak GPU Memory:** ~18-20GB
- **System RAM:** ~8-12GB
- **Disk Space:** ~30-40GB for results

## 🚀 Success Indicators

✅ **Setup Complete When:**
- `nvidia-smi` shows your GPU
- Conda environment activates without errors
- PyTorch detects CUDA
- Streamlit interface loads at `http://<ip>:8501`

✅ **Evaluation Running When:**
- Progress bars advance in Streamlit
- GPU utilization >80% in monitor
- Log files update in real-time
- No error messages in console

✅ **Evaluation Complete When:**
- All benchmarks show "Completed" status
- Results files generated in `/workspace/hierragmed/evaluation/results/`
- Final report available for download
- GPU utilization returns to idle

## 📞 Support

If you encounter issues:
1. Check troubleshooting section above
2. Review log files in `/workspace/hierragmed/evaluation/logs/`
3. Verify GPU memory availability with `nvidia-smi`
4. Restart evaluation from last checkpoint if needed

**No Docker required** - everything runs directly on the RunPod instance with conda environments!
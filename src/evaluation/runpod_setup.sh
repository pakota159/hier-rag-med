#!/bin/bash
# RunPod Environment Setup Script for HierRAGMed GPU Evaluation
# Optimized for RTX 4090 and RunPod infrastructure - CORRECTED VERSION

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
WORKSPACE_DIR="/workspace"
PROJECT_NAME="hierragmed"
PROJECT_DIR="${WORKSPACE_DIR}/${PROJECT_NAME}"
CONDA_ENV_NAME="hierragmed-gpu"
PYTHON_VERSION="3.10"

# Check if running on RunPod
check_runpod_environment() {
    log_info "Checking RunPod environment..."
    
    if [ ! -d "/workspace" ]; then
        log_error "Not running on RunPod! /workspace directory not found."
        exit 1
    fi
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "NVIDIA drivers not found! GPU required for evaluation."
        exit 1
    fi
    
    log_success "RunPod environment detected"
}

# Display system information
display_system_info() {
    log_info "System Information:"
    echo "===================="
    
    # GPU Information
    log_info "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    
    # CUDA Information
    if command -v nvcc &> /dev/null; then
        log_info "CUDA Version: $(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
    else
        log_warning "NVCC not found in PATH"
    fi
    
    # Python Information
    log_info "Python Version: $(python3 --version)"
    
    # Disk Space
    log_info "Disk Space:"
    df -h /workspace
    
    echo "===================="
}

# Setup directories - FINAL CORRECTED VERSION
setup_directories() {
    log_info "Setting up directory structure..."
    
    # The main project directory should already exist from git clone
    # Verify project structure exists
    if [ ! -d "${PROJECT_DIR}/src" ]; then
        log_error "Source code not found! Make sure you uploaded HierRAGMed to ${PROJECT_DIR}"
        echo "Expected structure:"
        echo "  ${PROJECT_DIR}/src/"
        echo "  ${PROJECT_DIR}/config.yaml"
        echo "  ${PROJECT_DIR}/requirements.txt"
        echo ""
        echo "Available directories in /workspace:"
        ls -la /workspace/ || true
        exit 1
    fi
    
    # Create data directories that match README.md structure
    log_info "Creating data directories..."
    mkdir -p "${PROJECT_DIR}/data/raw"
    mkdir -p "${PROJECT_DIR}/data/kg_raw" 
    mkdir -p "${PROJECT_DIR}/data/foundation_dataset"
    mkdir -p "${PROJECT_DIR}/data/processed"
    mkdir -p "${PROJECT_DIR}/data/vector_db"
    mkdir -p "${PROJECT_DIR}/data/logs"
    mkdir -p "${PROJECT_DIR}/data/benchmarks"
    
    # Create evaluation directories that match existing config structure
    log_info "Creating evaluation directories..."
    mkdir -p "${PROJECT_DIR}/evaluation"
    mkdir -p "${PROJECT_DIR}/evaluation/results" 
    mkdir -p "${PROJECT_DIR}/evaluation/cache"
    mkdir -p "${PROJECT_DIR}/evaluation/logs"
    
    # Create Streamlit config directory
    mkdir -p "${PROJECT_DIR}/.streamlit"
    
    # Create logs directory at root level (matches .gitignore)
    mkdir -p "${PROJECT_DIR}/logs"
    
    # Set permissions only for created directories
    chmod -R 755 "${PROJECT_DIR}/data" 2>/dev/null || true
    chmod -R 755 "${PROJECT_DIR}/evaluation" 2>/dev/null || true
    chmod -R 755 "${PROJECT_DIR}/.streamlit" 2>/dev/null || true
    chmod -R 755 "${PROJECT_DIR}/logs" 2>/dev/null || true
    
    # Verify critical project files exist
    log_info "Verifying project structure..."
    
    critical_files=(
        "src/evaluation"
        "src/basic_reasoning" 
        "config.yaml"
        "requirements.txt"
    )
    
    for file in "${critical_files[@]}"; do
        if [ -e "${PROJECT_DIR}/${file}" ]; then
            log_success "✅ Found: ${file}"
        else
            log_warning "⚠️  Missing: ${file}"
        fi
    done
    
    # Show final directory structure
    log_info "Final directory structure:"
    if command -v tree &> /dev/null; then
        tree "${PROJECT_DIR}" -L 3 -d 2>/dev/null || true
    else
        find "${PROJECT_DIR}" -type d -maxdepth 3 | head -20
    fi
    
    log_success "Directory structure setup complete"
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    # Update package list
    apt-get update -qq
    
    # Install essential packages
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        build-essential \
        software-properties-common \
        ca-certificates \
        gnupg \
        lsb-release \
        htop \
        nvtop \
        tmux \
        vim \
        tree \
        unzip
    
    # Clean up
    apt-get clean
    rm -rf /var/lib/apt/lists/*
    
    log_success "System dependencies installed"
}

# Setup Conda environment
setup_conda_environment() {
    log_info "Setting up Conda environment..."
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        log_info "Installing Miniconda..."
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p /opt/miniconda
        export PATH="/opt/miniconda/bin:$PATH"
        echo 'export PATH="/opt/miniconda/bin:$PATH"' >> ~/.bashrc
        rm /tmp/miniconda.sh
    fi
    
    # Initialize conda
    source /opt/miniconda/etc/profile.d/conda.sh
    
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "${CONDA_ENV_NAME}"; then
        log_info "Creating Conda environment: ${CONDA_ENV_NAME}"
        conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi
    
    # Activate environment
    conda activate "${CONDA_ENV_NAME}"
    
    log_success "Conda environment ready: ${CONDA_ENV_NAME}"
}

# Install PyTorch with CUDA support
install_pytorch() {
    log_info "Installing PyTorch with CUDA support..."
    
    # Activate conda environment
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV_NAME}"
    
    # Install PyTorch with CUDA 11.8 (compatible with RTX 4090)
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # Verify PyTorch installation
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
    
    log_success "PyTorch with CUDA support installed"
}

# Install Python dependencies - CORRECTED VERSION  
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Activate conda environment
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV_NAME}"
    
    # Use the existing requirements.txt from the project
    if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
        log_info "Using existing requirements.txt"
        pip install --no-cache-dir -r "${PROJECT_DIR}/requirements.txt"
    else
        log_warning "requirements.txt not found, creating minimal GPU requirements"
        # Fallback requirements if file doesn't exist
        cat > "${PROJECT_DIR}/requirements_runpod_fallback.txt" << 'EOF'
# Minimal GPU requirements for HierRAGMed evaluation
torch>=2.0.0
transformers>=4.25.0
sentence-transformers>=2.2.0
streamlit>=1.28.0
chromadb>=0.4.0
pandas>=1.5.0
numpy>=1.24.0
loguru>=0.6.0
pyyaml>=6.0
requests>=2.28.0
ollama>=0.1.0
accelerate>=0.16.0
datasets>=2.8.0
plotly>=5.12.0
scikit-learn>=1.2.0
tqdm>=4.64.0
httpx>=0.23.0
aiohttp>=3.8.0
faiss-cpu>=1.7.0
nltk>=3.8.0
huggingface-hub>=0.12.0
tokenizers>=0.13.0
scipy>=1.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
rich>=13.0.0
typer>=0.7.0
pydantic>=1.10.0
psutil>=5.9.0
EOF
        pip install --no-cache-dir -r "${PROJECT_DIR}/requirements_runpod_fallback.txt"
    fi
    
    # Install additional GPU-specific packages
    pip install --no-cache-dir \
        nvidia-ml-py3 \
        py3nvml \
        pynvml \
        gpustat
    
    log_success "Python dependencies installed"
}

# Setup Ollama for local LLM inference
setup_ollama() {
    log_info "Setting up Ollama for local LLM inference..."
    
    # Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # Start Ollama service in background
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to start
    sleep 10
    
    # Pull the required model
    log_info "Downloading Mistral 7B model (this may take a few minutes)..."
    ollama pull mistral:7b-instruct
    
    # Verify Ollama installation
    if ollama list | grep -q "mistral:7b-instruct"; then
        log_success "Ollama and Mistral model ready"
    else
        log_warning "Ollama model download may have failed"
    fi
}

# Configure CUDA environment
configure_cuda_environment() {
    log_info "Configuring CUDA environment..."
    
    # Set CUDA environment variables
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=8
    
    # Add to bashrc for persistence
    cat >> ~/.bashrc << 'EOF'

# CUDA Configuration for HierRAGMed
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
EOF
    
    log_success "CUDA environment configured"
}

# Create Streamlit config - CORRECTED VERSION
create_streamlit_config() {
    log_info "Creating Streamlit configuration..."
    
    # Check if project has existing streamlit config
    if [ -f "${PROJECT_DIR}/.streamlit/config.toml" ]; then
        log_info "Using existing Streamlit configuration"
        return
    fi
    
    # Create streamlit config for RunPod
    cat > "${PROJECT_DIR}/.streamlit/config.toml" << 'EOF'
[server]
address = "0.0.0.0"
port = 8501
enableXsrfProtection = false
enableCORS = true

[theme]
base = "dark"
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false

[client]
caching = true

[global]
maxUploadSize = 200
maxMessageSize = 200
EOF
    
    log_success "Streamlit configuration created"
}

# Download sample data (placeholder)
download_sample_data() {
    log_info "Preparing sample data..."
    
    # Create data directory structure
    mkdir -p "${PROJECT_DIR}/data/benchmarks"
    
    # Create placeholder files (actual datasets will be auto-downloaded during evaluation)
    echo '{"samples": [], "metadata": {"name": "MIRAGE", "version": "1.0", "note": "Will be auto-downloaded during evaluation"}}' > "${PROJECT_DIR}/data/benchmarks/mirage_placeholder.json"
    echo '{"samples": [], "metadata": {"name": "MedReason", "version": "1.0", "note": "Will be auto-downloaded during evaluation"}}' > "${PROJECT_DIR}/data/benchmarks/medreason_placeholder.json"
    
    log_success "Sample data prepared (actual datasets will be auto-downloaded)"
}

# Create startup scripts - CORRECTED VERSION
create_startup_scripts() {
    log_info "Creating startup scripts..."
    
    # Main startup script - Updated to use existing evaluation structure
    cat > "${PROJECT_DIR}/start_evaluation.sh" << 'EOF'
#!/bin/bash
# HierRAGMed Evaluation Startup Script

echo "🚀 Starting HierRAGMed GPU Evaluation"
echo "====================================="

# Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hierragmed-gpu

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Start Ollama service if not running
if ! pgrep -f ollama > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 5
fi

# Verify Ollama is working
if ! ollama list > /dev/null 2>&1; then
    echo "❌ Ollama service not responding"
    exit 1
fi

# Navigate to project directory
cd /workspace/hierragmed

echo "📊 Available Streamlit applications:"
find src/ -name "streamlit*.py" -o -name "*app.py" | sort

# Start Streamlit evaluation app (priority order)
if [ -f "src/evaluation/streamlit_evaluation.py" ]; then
    echo "🎯 Starting GPU evaluation interface..."
    streamlit run src/evaluation/streamlit_evaluation.py --server.port 8501 --server.address 0.0.0.0
elif [ -f "src/basic_reasoning/streamlit_app.py" ]; then
    echo "🧠 Starting basic reasoning interface..."
    streamlit run src/basic_reasoning/streamlit_app.py --server.port 8503 --server.address 0.0.0.0  
elif [ -f "src/kg/streamlit_app.py" ]; then
    echo "🔗 Starting KG interface..."
    streamlit run src/kg/streamlit_app.py --server.port 8502 --server.address 0.0.0.0
elif [ -f "src/simple/streamlit_app.py" ]; then
    echo "✨ Starting simple interface..."
    streamlit run src/simple/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
else
    echo "❌ No Streamlit app found!"
    echo "Available Python files in src/:"
    find src/ -name "*.py" | head -10
    exit 1
fi
EOF
    
    # Make executable
    chmod +x "${PROJECT_DIR}/start_evaluation.sh"
    
    # GPU monitoring script
    cat > "${PROJECT_DIR}/monitor_gpu.sh" << 'EOF'
#!/bin/bash
# GPU Monitoring Script

echo "🔥 GPU Monitoring - Press Ctrl+C to stop"
echo "========================================="

while true; do
    clear
    echo "$(date)"
    echo "========================================="
    nvidia-smi
    echo ""
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "Used: %s MB / %s MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
    echo "========================================="
    sleep 2
done
EOF
    
    chmod +x "${PROJECT_DIR}/monitor_gpu.sh"
    
    # Alternative evaluation runner script
    cat > "${PROJECT_DIR}/run_evaluation_direct.sh" << 'EOF'
#!/bin/bash
# Direct evaluation runner (command line)

echo "🧪 Starting Direct Evaluation"
echo "=============================="

# Activate conda environment  
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hierragmed-gpu

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

cd /workspace/hierragmed

# Check if evaluation runner exists
if [ -f "src/evaluation/run_evaluation.py" ]; then
    echo "🎯 Running GPU evaluation directly..."
    python src/evaluation/run_evaluation.py
elif [ -f "src/evaluation/compare_models.py" ]; then
    echo "📊 Running model comparison..."
    python src/evaluation/compare_models.py
else
    echo "❌ No evaluation runner found!"
    echo "Available evaluation scripts:"
    find src/evaluation/ -name "*.py" -type f | head -10
    exit 1
fi
EOF

    chmod +x "${PROJECT_DIR}/run_evaluation_direct.sh"
    
    log_success "Startup scripts created"
}

# Verify installation - CORRECTED VERSION
verify_installation() {
    log_info "Verifying installation..."
    
    # Activate conda environment
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV_NAME}"
    
    # Test PyTorch CUDA
    python -c "
import torch
import sys
import os

print('=' * 60)
print('HIERRAGMED GPU EVALUATION VERIFICATION')
print('=' * 60)

# Check project structure
project_dir = '/workspace/hierragmed'
print(f'Project directory: {project_dir}')
print(f'Project exists: {os.path.exists(project_dir)}')

if os.path.exists(project_dir):
    print('\\nProject structure:')
    for item in ['src/', 'data/', 'config.yaml', 'requirements.txt']:
        path = os.path.join(project_dir, item)
        exists = os.path.exists(path)
        print(f'  {item}: {\"✅\" if exists else \"❌\"}')

# PyTorch verification
print(f'\\nPyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.current_device()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    
    # Memory info
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU memory: {memory_total:.1f} GB')
    
    # Test tensor operations
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        print('✅ GPU tensor operations working')
        del x, y  # Clean up
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'❌ GPU tensor operations failed: {e}')
        sys.exit(1)
else:
    print('❌ CUDA not available')
    sys.exit(1)

# Test key packages
packages_to_test = [
    ('transformers', 'Transformers'),
    ('streamlit', 'Streamlit'), 
    ('sentence_transformers', 'Sentence Transformers'),
    ('chromadb', 'ChromaDB'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('ollama', 'Ollama'),
    ('nvidia_ml_py3', 'NVIDIA ML')
]

for package_name, display_name in packages_to_test:
    try:
        package = __import__(package_name)
        version = getattr(package, '__version__', 'unknown')
        print(f'✅ {display_name}: {version}')
    except ImportError as e:
        print(f'⚠️ {display_name} import failed: {e}')

print('=' * 60)
print('🚀 INSTALLATION VERIFICATION COMPLETE')
print('=' * 60)
"
    
    if [ $? -eq 0 ]; then
        log_success "Installation verification passed!"
    else
        log_error "Installation verification failed!"
        exit 1
    fi
}

# Main setup function
main() {
    echo "🚀 HierRAGMed RunPod Setup Script"
    echo "=================================="
    echo "Optimized for RTX 4090 GPU evaluation"
    echo ""
    
    # Run setup steps
    check_runpod_environment
    display_system_info
    setup_directories
    install_system_dependencies
    setup_conda_environment
    install_pytorch
    install_python_dependencies
    setup_ollama
    configure_cuda_environment
    create_streamlit_config
    download_sample_data
    create_startup_scripts
    verify_installation
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "=================================="
    echo ""
    echo "📋 Next steps:"
    echo "1. Your HierRAGMed source code is at: ${PROJECT_DIR}"
    echo "2. Start evaluation with: ${PROJECT_DIR}/start_evaluation.sh"
    echo "3. Monitor GPU usage with: ${PROJECT_DIR}/monitor_gpu.sh"
    echo "4. Direct evaluation: ${PROJECT_DIR}/run_evaluation_direct.sh"
    echo "5. Access Streamlit UI at: http://<runpod-ip>:8501"
    echo ""
    echo "🔥 Environment details:"
    echo "   Conda environment: ${CONDA_ENV_NAME}"
    echo "   Project directory: ${PROJECT_DIR}"
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo ""
    echo "✅ Ready for GPU evaluation!"
}

# Run main function
main "$@"
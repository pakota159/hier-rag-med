#!/bin/bash
# RunPod Environment Setup Script for HierRAGMed GPU Evaluation
# Optimized for RTX 4090 and RunPod infrastructure

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

# Setup directories
setup_directories() {
    log_info "Setting up directory structure..."
    
    # Create main directories
    mkdir -p "${PROJECT_DIR}"
    mkdir -p "${PROJECT_DIR}/data"
    mkdir -p "${PROJECT_DIR}/evaluation/results"
    mkdir -p "${PROJECT_DIR}/evaluation/cache"
    mkdir -p "${PROJECT_DIR}/evaluation/logs"
    mkdir -p "${PROJECT_DIR}/evaluation/configs"
    mkdir -p "${PROJECT_DIR}/.streamlit"
    
    # Set permissions
    chmod -R 755 "${PROJECT_DIR}"
    
    log_success "Directory structure created"
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
        tree
    
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

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Activate conda environment
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV_NAME}"
    
    # Create requirements file if not exists
    cat > "${PROJECT_DIR}/requirements_runpod.txt" << 'EOF'
# GPU-optimized requirements for RunPod evaluation
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.25.0
accelerate>=0.16.0
datasets>=2.8.0
sentence-transformers>=2.2.0
streamlit>=1.28.0
plotly>=5.12.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
loguru>=0.6.0
pyyaml>=6.0
tqdm>=4.64.0
httpx>=0.23.0
aiohttp>=3.8.0
requests>=2.28.0
chromadb>=0.4.0
faiss-cpu>=1.7.0
nltk>=3.8.0
spacy>=3.4.0
gensim>=4.2.0
openai>=0.26.0
anthropic>=0.7.0
ollama>=0.1.0
huggingface-hub>=0.12.0
tokenizers>=0.13.0
scipy>=1.10.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.20.0
rich>=13.0.0
typer>=0.7.0
pydantic>=1.10.0
sqlalchemy>=1.4.0
alembic>=1.9.0
psutil>=5.9.0
memory-profiler>=0.60.0
line-profiler>=4.0.0
EOF
    
    # Install dependencies
    pip install --no-cache-dir -r "${PROJECT_DIR}/requirements_runpod.txt"
    
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
    
    # Wait for service to start
    sleep 5
    
    # Pull required models
    log_info "Pulling Mistral 7B model..."
    ollama pull mistral:7b-instruct
    
    # Optional: Pull additional models
    # ollama pull llama2:7b-chat
    # ollama pull codellama:7b-instruct
    
    log_success "Ollama setup completed"
}

# Configure CUDA environment
configure_cuda_environment() {
    log_info "Configuring CUDA environment..."
    
    # Set CUDA environment variables
    cat >> ~/.bashrc << 'EOF'

# CUDA Environment Variables for RTX 4090 Optimization
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMBA_CACHE_DIR=/tmp/numba_cache
EOF
    
    # Create CUDA cache directory
    mkdir -p /tmp/numba_cache
    
    # Source the environment
    source ~/.bashrc
    
    log_success "CUDA environment configured"
}

# Create Streamlit configuration
create_streamlit_config() {
    log_info "Creating Streamlit configuration..."
    
    # Create .streamlit directory
    mkdir -p "${PROJECT_DIR}/.streamlit"
    
    # Create config.toml
    cat > "${PROJECT_DIR}/.streamlit/config.toml" << 'EOF'
[server]
address = "0.0.0.0"
port = 8501
enableXsrfProtection = false
enableCORS = true
maxUploadSize = 200
maxMessageSize = 200

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
installTracer = false

[logger]
level = "info"

[client]
caching = true
displayEnabled = true

[global]
developmentMode = false
EOF
    
    log_success "Streamlit configuration created"
}

# Download sample data (optional)
download_sample_data() {
    log_info "Downloading sample evaluation data..."
    
    # Create data directory structure
    mkdir -p "${PROJECT_DIR}/data/benchmarks"
    
    # Download sample datasets (replace with actual URLs)
    # wget -q -O "${PROJECT_DIR}/data/benchmarks/mirage_sample.json" "https://example.com/mirage_sample.json"
    # wget -q -O "${PROJECT_DIR}/data/benchmarks/medreason_sample.json" "https://example.com/medreason_sample.json"
    
    # Create placeholder files for now
    echo '{"samples": [], "metadata": {"name": "MIRAGE", "version": "1.0"}}' > "${PROJECT_DIR}/data/benchmarks/mirage_sample.json"
    echo '{"samples": [], "metadata": {"name": "MedReason", "version": "1.0"}}' > "${PROJECT_DIR}/data/benchmarks/medreason_sample.json"
    
    log_success "Sample data prepared"
}

# Create startup scripts
create_startup_scripts() {
    log_info "Creating startup scripts..."
    
    # Main startup script
    cat > "${PROJECT_DIR}/start_evaluation.sh" << 'EOF'
#!/bin/bash
# HierRAGMed Evaluation Startup Script

# Activate conda environment
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hierragmed-gpu

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start Ollama service
ollama serve &

# Wait for Ollama to start
sleep 5

# Start Streamlit app
cd /workspace/hierragmed
streamlit run src/evaluation/streamlit_evaluation.py --server.port 8501 --server.address 0.0.0.0
EOF
    
    # Make executable
    chmod +x "${PROJECT_DIR}/start_evaluation.sh"
    
    # GPU monitoring script
    cat > "${PROJECT_DIR}/monitor_gpu.sh" << 'EOF'
#!/bin/bash
# GPU Monitoring Script

echo "GPU Monitoring - Press Ctrl+C to stop"
echo "====================================="

while true; do
    clear
    echo "$(date)"
    echo "====================================="
    nvidia-smi
    echo "====================================="
    sleep 2
done
EOF
    
    chmod +x "${PROJECT_DIR}/monitor_gpu.sh"
    
    log_success "Startup scripts created"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Activate conda environment
    source /opt/miniconda/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV_NAME}"
    
    # Test PyTorch CUDA
    python -c "
import torch
import sys

print('=' * 50)
print('HIERRAGMED GPU EVALUATION VERIFICATION')
print('=' * 50)

# PyTorch
print(f'PyTorch version: {torch.__version__}')
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
    except Exception as e:
        print(f'❌ GPU tensor operations failed: {e}')
        sys.exit(1)
else:
    print('❌ CUDA not available')
    sys.exit(1)

# Test other packages
try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers import failed: {e}')

try:
    import streamlit
    print(f'✅ Streamlit: {streamlit.__version__}')
except ImportError as e:
    print(f'❌ Streamlit import failed: {e}')

try:
    import sentence_transformers
    print(f'✅ Sentence Transformers: {sentence_transformers.__version__}')
except ImportError as e:
    print(f'❌ Sentence Transformers import failed: {e}')

print('=' * 50)
print('🚀 INSTALLATION VERIFICATION COMPLETE')
print('=' * 50)
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
    echo "Next steps:"
    echo "1. Upload your HierRAGMed source code to ${PROJECT_DIR}"
    echo "2. Run the evaluation with: ${PROJECT_DIR}/start_evaluation.sh"
    echo "3. Monitor GPU usage with: ${PROJECT_DIR}/monitor_gpu.sh"
    echo "4. Access Streamlit UI at: http://<runpod-ip>:8501"
    echo ""
    echo "Environment activated: ${CONDA_ENV_NAME}"
    echo "Project directory: ${PROJECT_DIR}"
    echo ""
}

# Run main function
main "$@"
#!/bin/bash

# KeyBot Setup and Run Script
# This script clones the repository, sets up the environment, and runs the project

set -e  # Exit on error

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GITHUB_REPO_URL="${GITHUB_REPO_URL:-https://github.com/YOUR_USERNAME/keybot.git}"
PROJECT_NAME="keybot_project"

# Determine project directory
# If we're already in a keybot directory (has pyproject.toml), use current dir
# Otherwise use ~/keybot_project for cloning
if [ -f "pyproject.toml" ] && [ -f "setup_and_run.sh" ]; then
    PROJECT_DIR="$(pwd)"
else
    PROJECT_DIR="${HOME}/${PROJECT_NAME}"
fi

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_section() {
    echo -e "\n${BLUE}================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ========================================
# 1. Clone Repository
# ========================================
clone_repository() {
    print_section "1. Cloning Repository"
    
    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Repository already exists at $PROJECT_DIR"
        read -p "Do you want to pull latest changes? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd "$PROJECT_DIR"
            git pull
            print_success "Repository updated"
        fi
    else
        print_info "Cloning from: $GITHUB_REPO_URL"
        git clone "$GITHUB_REPO_URL" "$PROJECT_DIR"
        print_success "Repository cloned to $PROJECT_DIR"
    fi
    
    cd "$PROJECT_DIR"
}

# ========================================
# 2. Setup Environment
# ========================================
setup_environment() {
    print_section "2. Setting Up Environment"
    
    # Add Poetry to PATH if it exists
    export PATH="$HOME/.local/bin:$PATH"
    
    # Check for Poetry
    if ! command_exists poetry; then
        print_error "Poetry is not installed"
        print_info "Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
    else
        print_success "Poetry is installed: $(poetry --version)"
    fi
    
    # Setup Python version with pyenv if available
    if command_exists pyenv; then
        print_info "Setting Python version to 3.12.6 using pyenv"
        pyenv local 3.12.6 || print_warning "Python 3.12.6 not available in pyenv, using system Python"
        
        # Get Python path and configure Poetry
        if pyenv which python >/dev/null 2>&1; then
            PYTHON_PATH=$(pyenv which python)
            poetry env use "$PYTHON_PATH"
            print_success "Poetry configured to use: $PYTHON_PATH"
        fi
    else
        print_warning "pyenv not available, using system Python: $(python3 --version)"
    fi
    
    # Install dependencies
    print_info "Installing dependencies (this may take several minutes)..."
    poetry install
    print_success "All dependencies installed"
}

# ========================================
# 3. Verify Installation
# ========================================
verify_installation() {
    print_section "3. Verifying Installation"
    
    cd "$PROJECT_DIR"
    poetry run python -c "
import torch
import cv2
import pandas
import albumentations

print('✓ PyTorch:', torch.__version__)
print('✓ OpenCV:', cv2.__version__)
print('✓ Pandas:', pandas.__version__)
print('✓ Albumentations:', albumentations.__version__)

if torch.cuda.is_available():
    print('✓ CUDA available:', torch.cuda.get_device_name(0))
else:
    print('⚠ CUDA not available - will use CPU (slower)')
"
    
    print_success "Installation verified"
}

# ========================================
# 4. Prepare Data
# ========================================
prepare_data() {
    print_section "4. Data Preparation"
    
    RAW_DATA_DIR="${PROJECT_DIR}/codes/preprocess_data/AASCE_rawdata/boostnet_labeldata"
    mkdir -p "$RAW_DATA_DIR"
    
    print_success "Created data directory: $RAW_DATA_DIR"
    
    # Check if data exists
    if [ -z "$(ls -A $RAW_DATA_DIR 2>/dev/null)" ]; then
        print_warning "Dataset not found!"
        print_info "Please download the AASCE dataset from:"
        print_info "http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images"
        print_info "Extract and place the data in: $RAW_DATA_DIR"
        
        read -p "Press Enter after you have downloaded and extracted the dataset, or 's' to skip: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            print_warning "Skipping data preparation"
            return
        fi
    else
        print_success "Dataset found"
    fi
    
    # Run preprocessing
    print_info "Running data preprocessing..."
    cd "${PROJECT_DIR}/codes/preprocess_data"
    poetry run python preprocess_data.py
    cd "$PROJECT_DIR"
    print_success "Data preprocessing completed"
}

# ========================================
# 5. Train Model
# ========================================
train_model() {
    print_section "5. Training Model"
    
    read -p "Start training? This will take a long time (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Skipping training"
        return
    fi
    
    cd "${PROJECT_DIR}/codes"
    
    # Train interactive keypoint model (if script exists)
    if [ -f "train_interactive_keypoint_model.sh" ]; then
        print_info "Training interactive keypoint model..."
        bash train_interactive_keypoint_model.sh
    fi
    
    # Train AASCE model
    print_info "Training AASCE model..."
    poetry run python train_AASCE.py
    
    cd "$PROJECT_DIR"
    print_success "Training completed"
}

# ========================================
# 6. Run Evaluation
# ========================================
run_evaluation() {
    print_section "6. Running Evaluation"
    
    read -p "Run evaluation? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Skipping evaluation"
        return
    fi
    
    cd "${PROJECT_DIR}/codes"
    print_info "Running evaluation..."
    poetry run python evaluate_AASCE.py
    cd "$PROJECT_DIR"
    print_success "Evaluation completed"
}

# ========================================
# 7. Show Results
# ========================================
show_results() {
    print_section "7. Results"
    
    SAVE_DIR="${PROJECT_DIR}/save/AASCE_interactive_keypoint_estimation"
    
    if [ -d "$SAVE_DIR" ]; then
        print_info "Saved files:"
        ls -lh "$SAVE_DIR" | tail -n +2 | awk '{printf "  - %s (%s)\n", $9, $5}'
        
        # Show last 30 lines of log
        LOG_FILE="${SAVE_DIR}/log.txt"
        if [ -f "$LOG_FILE" ]; then
            echo
            print_info "Training log (last 30 lines):"
            tail -n 30 "$LOG_FILE"
        fi
    else
        print_warning "No saved files found yet"
    fi
}

# ========================================
# 8. Start TensorBoard
# ========================================
start_tensorboard() {
    print_section "8. TensorBoard"
    
    read -p "Start TensorBoard? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi
    
    SAVE_DIR="${PROJECT_DIR}/save/AASCE_interactive_keypoint_estimation"
    
    if [ -d "$SAVE_DIR" ]; then
        print_info "Starting TensorBoard on http://localhost:6006"
        print_info "Press Ctrl+C to stop"
        cd "$PROJECT_DIR"
        poetry run tensorboard --logdir "$SAVE_DIR" --port 6006
    else
        print_warning "No training data found for TensorBoard"
    fi
}

# ========================================
# Main Menu
# ========================================
show_menu() {
    echo
    print_section "KeyBot Setup and Run Script"
    echo "1. Clone repository"
    echo "2. Setup environment"
    echo "3. Verify installation"
    echo "4. Prepare data"
    echo "5. Train model"
    echo "6. Run evaluation"
    echo "7. Show results"
    echo "8. Start TensorBoard"
    echo "9. Run all (1-6)"
    echo "0. Exit"
    echo
}

run_all() {
    clone_repository
    setup_environment
    verify_installation
    prepare_data
    train_model
    run_evaluation
    show_results
}

# ========================================
# Main Script
# ========================================
main() {
    echo -e "${BLUE}"
    cat << "EOF"
    ██╗  ██╗███████╗██╗   ██╗██████╗  ██████╗ ████████╗
    ██║ ██╔╝██╔════╝╚██╗ ██╔╝██╔══██╗██╔═══██╗╚══██╔══╝
    █████╔╝ █████╗   ╚████╔╝ ██████╔╝██║   ██║   ██║   
    ██╔═██╗ ██╔══╝    ╚██╔╝  ██╔══██╗██║   ██║   ██║   
    ██║  ██╗███████╗   ██║   ██████╔╝╚██████╔╝   ██║   
    ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═════╝  ╚═════╝    ╚═╝   
EOF
    echo -e "${NC}"
    echo "Vertebrae Keypoint Estimation Setup Script"
    echo "ECCV 2024"
    echo
    
    # Check if we're running with arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice
            case $choice in
                1) clone_repository ;;
                2) setup_environment ;;
                3) verify_installation ;;
                4) prepare_data ;;
                5) train_model ;;
                6) run_evaluation ;;
                7) show_results ;;
                8) start_tensorboard ;;
                9) run_all ;;
                0) print_info "Exiting..."; exit 0 ;;
                *) print_error "Invalid option" ;;
            esac
            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case $1 in
            clone) clone_repository ;;
            setup) setup_environment ;;
            verify) verify_installation ;;
            data) prepare_data ;;
            train) train_model ;;
            eval) run_evaluation ;;
            results) show_results ;;
            tensorboard) start_tensorboard ;;
            all) run_all ;;
            *)
                echo "Usage: $0 [clone|setup|verify|data|train|eval|results|tensorboard|all]"
                echo "Or run without arguments for interactive menu"
                exit 1
                ;;
        esac
    fi
}

# Run main function
main "$@"


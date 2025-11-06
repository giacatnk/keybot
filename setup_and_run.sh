#!/bin/bash

# KeyBot Setup and Run Script
# This script clones the repository, sets up the environment, and runs the project
# Works on: Local machine, Google Colab

set -e  # Exit on error

# Detect if running in Google Colab
IS_COLAB=false
if [ -d "/content" ] && [ -f "/usr/local/lib/python3.10/dist-packages/google/colab/_ipython.py" ] 2>/dev/null || [ -n "$COLAB_GPU" ]; then
    IS_COLAB=true
    echo "🔍 Detected Google Colab environment"
fi

# Add Poetry to PATH (only needed for local)
export PATH="$HOME/.local/bin:$PATH"

# Non-interactive mode flag
NON_INTERACTIVE=false
if [[ "$*" == *"-y"* ]] || [[ "$*" == *"--yes"* ]] || [ "$IS_COLAB" = true ]; then
    NON_INTERACTIVE=true
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GITHUB_REPO_URL="${GITHUB_REPO_URL:-https://github.com/giacatnk/keybot.git}"
PROJECT_NAME="keybot_project"

# Use token for authentication if available
if [ ! -z "$GITHUB_TOKEN" ]; then
    # Replace https:// with https://token@
    GITHUB_REPO_URL="${GITHUB_REPO_URL/https:\/\//https:\/\/${GITHUB_TOKEN}@}"
fi

# Determine project directory
# If we're already in a keybot directory (has pyproject.toml), use current dir
# Otherwise use ~/keybot_project for cloning
if [ -f "pyproject.toml" ] && [ -f "setup_and_run.sh" ]; then
    PROJECT_DIR="$(pwd)"
elif [ "$IS_COLAB" = true ]; then
    PROJECT_DIR="/content/keybot"
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
    
    # Configure git if token is available
    if [ ! -z "$GITHUB_TOKEN" ]; then
        git config --global credential.helper store
        print_info "Using GitHub token for authentication"
    fi
    
    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Repository already exists at $PROJECT_DIR"
        if [ "$NON_INTERACTIVE" = true ]; then
            cd "$PROJECT_DIR"
            git pull
            print_success "Repository updated"
        else
            read -p "Do you want to pull latest changes? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                cd "$PROJECT_DIR"
                git pull
                print_success "Repository updated"
            fi
        fi
    else
        print_info "Cloning from: https://github.com/giacatnk/keybot.git"
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
    
    if [ "$IS_COLAB" = true ]; then
        # Google Colab setup - use pip
        print_info "Setting up for Google Colab (using pip)..."
        print_info "Python version: $(python3 --version)"
        
        # Install dependencies from requirements.txt
        print_info "Installing dependencies..."
        pip install -q -r requirements.txt 2>/dev/null || {
            print_warning "requirements.txt not found, installing manually..."
            pip install -q torch torchvision albumentations scipy munch pytz tqdm opencv-python pandas matplotlib
        }
        
        print_success "All dependencies installed (pip)"
        
    else
        # Local setup - use Poetry
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
        print_success "All dependencies installed (poetry)"
    fi
}

# ========================================
# 3. Verify Installation
# ========================================
verify_installation() {
    print_section "3. Verifying Installation"
    
    cd "$PROJECT_DIR"
    
    # Use python3 directly in Colab, poetry run python locally
    if [ "$IS_COLAB" = true ]; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="poetry run python"
    fi
    
    $PYTHON_CMD -c "
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
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ MPS (Apple Silicon) available')
else:
    print('⚠ CUDA/MPS not available - will use CPU (slower)')
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
        
        if [ "$NON_INTERACTIVE" = false ]; then
            read -p "Press Enter after you have downloaded and extracted the dataset, or 's' to skip: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Ss]$ ]]; then
                print_warning "Skipping data preparation"
                return
            fi
        else
            print_warning "Skipping data preparation in non-interactive mode"
            return
        fi
    else
        print_success "Dataset found"
    fi
    
    # Run preprocessing
    print_info "Running data preprocessing..."
    cd "${PROJECT_DIR}/codes/preprocess_data"
    
    # Use python3 directly in Colab, poetry run python locally
    if [ "$IS_COLAB" = true ]; then
        python3 data_split.py
        python3 preprocess_data.py
    else
        poetry run python data_split.py
        poetry run python preprocess_data.py
    fi
    
    cd "$PROJECT_DIR"
    print_success "Data preprocessing completed"
}

# ========================================
# 5. Train Model
# ========================================
train_model() {
    print_section "5. Training Model"
    
    # Check if dependencies are installed
    if [ "$IS_COLAB" = true ]; then
        CHECK_CMD="python3 -c"
    else
        CHECK_CMD="poetry run python -c"
    fi
    
    if ! $CHECK_CMD "import munch" 2>/dev/null; then
        print_warning "Dependencies not installed. Running setup first..."
        setup_environment
    fi
    
    if [ "$NON_INTERACTIVE" = false ]; then
        read -p "Start training? This will take a long time (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping training"
            return
        fi
    else
        print_info "Starting training in non-interactive mode..."
    fi
    
    cd "${PROJECT_DIR}/codes"
    
    # Set python command based on environment
    if [ "$IS_COLAB" = true ]; then
        PYTHON_CMD="python3"
        print_info "Training on Colab (this will take ~30 hours)..."
    else
        PYTHON_CMD="poetry run python"
    fi
    
    # Train interactive keypoint model
    if [ -f "train_interactive_keypoint_model.sh" ]; then
        print_info "Training interactive keypoint model (Refiner)..."
        # Run with appropriate environment (device auto-detected in code)
        CUDA_VISIBLE_DEVICES="0" $PYTHON_CMD -u main.py --seed 42 --config config_AASCE --save_test_prediction --subpixel_inference 15 --use_prev_heatmap_only_for_hint_index
    fi
    
    # Train AASCE model (Detector + Corrector)
    print_info "Training AASCE model (Detector + Corrector)..."
    $PYTHON_CMD train_AASCE.py
    
    cd "$PROJECT_DIR"
    print_success "Training completed"
}

# ========================================
# 6. Run Evaluation
# ========================================
run_evaluation() {
    print_section "6. Running Evaluation"
    
    # Check if dependencies are installed
    if [ "$IS_COLAB" = true ]; then
        CHECK_CMD="python3 -c"
    else
        CHECK_CMD="poetry run python -c"
    fi
    
    if ! $CHECK_CMD "import munch" 2>/dev/null; then
        print_warning "Dependencies not installed. Running setup first..."
        setup_environment
    fi
    
    if [ "$NON_INTERACTIVE" = false ]; then
        read -p "Run evaluation? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping evaluation"
            return
        fi
    else
        print_info "Running evaluation in non-interactive mode..."
    fi
    
    cd "${PROJECT_DIR}/codes"
    print_info "Running evaluation..."
    
    # Use python3 directly in Colab, poetry run python locally
    if [ "$IS_COLAB" = true ]; then
        python3 evaluate_AASCE.py
    else
        poetry run python evaluate_AASCE.py
    fi
    
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
# 7.5. Run Visualization
# ========================================
run_visualization() {
    print_section "7.5. Running Visualization"
    
    # Check if dependencies are installed
    if [ "$IS_COLAB" = true ]; then
        CHECK_CMD="python3 -c"
        PYTHON_CMD="python3"
    else
        CHECK_CMD="poetry run python -c"
        PYTHON_CMD="poetry run python"
    fi
    
    if ! $CHECK_CMD "import munch" 2>/dev/null; then
        print_warning "Dependencies not installed. Running setup first..."
        setup_environment
    fi
    
    # Check if models exist
    if [ ! -f "save/AASCE_interactive_keypoint_estimation/model.pth" ]; then
        print_error "Refiner model not found!"
        print_info "Please train or download the model first"
        return 1
    fi
    
    if [ ! -f "save_suggestion/AASCE_suggestModel.pth" ]; then
        print_warning "Detector model not found at save_suggestion/AASCE_suggestModel.pth"
    fi
    
    if [ ! -f "save_refine/AASCE_refineModel.pth" ]; then
        print_warning "Corrector model not found at save_refine/AASCE_refineModel.pth"
    fi
    
    cd "$PROJECT_DIR"
    
    # Always use vertebrae visualization (paper-style)
    print_info "Running vertebrae visualization (paper-style)..."
    $PYTHON_CMD visualize_vertebrae.py
    
    if [ -f "keybot_vertebrae_visualization.png" ]; then
        print_success "Visualization completed!"
        print_info "Output files:"
        print_info "  - keybot_vertebrae_visualization.png (paper-style)"
        [ -f "keybot_vertebrae_results.json" ] && print_info "  - keybot_vertebrae_results.json"
        echo
        print_info "For detailed keypoint visualization, run:"
        if [ "$IS_COLAB" = true ]; then
            echo "  !python3 load_and_visualize.py"
        else
            echo "  $PYTHON_CMD load_and_visualize.py"
        fi
        
        # Open image if on macOS
        if [[ "$OSTYPE" == "darwin"* ]] && [ "$NON_INTERACTIVE" = false ]; then
            read -p "Open visualization? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                open keybot_vertebrae_visualization.png
            fi
        fi
        
        # In Colab, show instructions
        if [ "$IS_COLAB" = true ]; then
            echo
            print_info "In Colab, view the image with:"
            echo "  from IPython.display import Image, display"
            echo "  display(Image('keybot_vertebrae_visualization.png'))"
        fi
    else
        print_error "Visualization failed - output file not created"
        return 1
    fi
}

# ========================================
# 8. Commit Models to GitHub
# ========================================
save_and_commit_models() {
    print_section "8. Committing Models to GitHub"
    
    SAVE_DIR="${PROJECT_DIR}/save/AASCE_interactive_keypoint_estimation"
    
    if [ ! -d "$SAVE_DIR" ]; then
        print_error "No trained models found at $SAVE_DIR"
        return 1
    fi
    
    # Configure git authentication if token is available
    if [ ! -z "$GITHUB_TOKEN" ]; then
        print_info "Configuring GitHub authentication with token..."
        git config --global credential.helper store
        
        # Update remote URL to use token
        git remote set-url origin "https://${GITHUB_TOKEN}@github.com/giacatnk/keybot.git"
        print_success "Authentication configured"
    fi
    
    # Show model files
    print_info "Model files to commit:"
    ls -lh "$SAVE_DIR" | tail -n +2 | awk '{printf "  - %s (%s)\n", $9, $5}'
    
    # Git operations
    print_info "Committing models to git..."
    
    # Add save directory and git attributes
    git add save/
    git add .gitattributes 2>/dev/null || true
    
    # Create commit with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    COMMIT_MSG="Add trained model checkpoint

Trained on: $(date '+%Y-%m-%d %H:%M:%S')
Timestamp: ${TIMESTAMP}
Location: save/AASCE_interactive_keypoint_estimation/"
    
    git commit -m "$COMMIT_MSG" || print_warning "No changes to commit"
    
    print_success "Models committed locally"
    
    # Ask to push to GitHub
    if [ "$NON_INTERACTIVE" = false ]; then
        read -p "Push models to GitHub? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Pushing to GitHub..."
            git push origin main
            print_success "Models pushed to GitHub!"
        else
            print_info "Skipped push. Run 'git push origin main' manually when ready."
        fi
    else
        print_info "Pushing to GitHub in non-interactive mode..."
        git push origin main
        print_success "Models pushed to GitHub!"
    fi
}

# ========================================
# 9. Start TensorBoard
# ========================================
start_tensorboard() {
    print_section "9. TensorBoard"
    
    if [ "$NON_INTERACTIVE" = false ]; then
        read -p "Start TensorBoard? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    else
        print_info "Starting TensorBoard in non-interactive mode..."
    fi
    
    SAVE_DIR="${PROJECT_DIR}/save/AASCE_interactive_keypoint_estimation"
    
    if [ -d "$SAVE_DIR" ]; then
        print_info "Starting TensorBoard on http://localhost:6006"
        print_info "Press Ctrl+C to stop"
        cd "$PROJECT_DIR"
        
        # Use appropriate command based on environment
        if [ "$IS_COLAB" = true ]; then
            # In Colab, use %load_ext tensorboard magic instead
            print_warning "In Colab, use this command in a notebook cell:"
            echo "  %load_ext tensorboard"
            echo "  %tensorboard --logdir $SAVE_DIR"
        else
            poetry run tensorboard --logdir "$SAVE_DIR" --port 6006
        fi
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
    echo "8. Run visualization (load_and_visualize.py)"
    echo "9. Commit models to GitHub"
    echo "10. Start TensorBoard"
    echo "11. Run all (1-6)"
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
    
    # Show Colab-specific instructions
    if [ "$IS_COLAB" = true ]; then
        print_info "Running in Google Colab mode"
        print_info "Usage in Colab notebook cell: !bash setup_and_run.sh <command>"
        echo
    fi
    
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
                8) run_visualization ;;
                9) save_and_commit_models ;;
                10) start_tensorboard ;;
                11) run_all ;;
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
            viz|visualize) run_visualization ;;
            save) save_and_commit_models ;;
            tensorboard) start_tensorboard ;;
            all) run_all ;;
            *)
                echo "Usage: $0 [clone|setup|verify|data|train|eval|results|viz|save|tensorboard|all]"
                echo "Or run without arguments for interactive menu"
                echo ""
                echo "Commands:"
                echo "  clone       - Clone the repository"
                echo "  setup       - Install dependencies"
                echo "  verify      - Verify installation"
                echo "  data        - Prepare dataset"
                echo "  train       - Train all models"
                echo "  eval        - Run evaluation"
                echo "  results     - Show training results"
                echo "  viz         - Run visualization (load_and_visualize.py)"
                echo "  save        - Commit models to GitHub"
                echo "  tensorboard - Start TensorBoard"
                echo "  all         - Run all steps"
                exit 1
                ;;
        esac
    fi
}

# Run main function
main "$@"


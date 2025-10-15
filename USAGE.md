# KeyBot Setup Script

## Quick Start

```bash
# Interactive mode (recommended)
./setup_and_run.sh

# Or run everything automatically
./setup_and_run.sh all
```

## Commands

```bash
./setup_and_run.sh clone        # Clone from GitHub
./setup_and_run.sh setup        # Install dependencies
./setup_and_run.sh verify       # Check installation
./setup_and_run.sh data         # Prepare dataset
./setup_and_run.sh train        # Train model
./setup_and_run.sh eval         # Run evaluation
./setup_and_run.sh results      # Show results
./setup_and_run.sh save         # Commit models to GitHub
./setup_and_run.sh tensorboard  # Start TensorBoard

# Non-interactive mode (for Google Colab or automation)
./setup_and_run.sh train -y     # Train without prompts
./setup_and_run.sh all --yes    # Run everything without prompts
```

## Configuration

Edit repository URL in `setup_and_run.sh` (line 13):
```bash
GITHUB_REPO_URL="https://github.com/YOUR_USERNAME/keybot.git"
```

## Requirements

- Git, Python 3.12+, Poetry (auto-installs)
- Download AASCE dataset: http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images
- Extract to: `~/keybot_project/codes/preprocess_data/AASCE_rawdata/boostnet_labeldata/`

## Troubleshooting

**Poetry not found:** Add to PATH: `export PATH="$HOME/.local/bin:$PATH"`

**Python version:** Install with pyenv: `pyenv install 3.12.6 && pyenv local 3.12.6`

**No CUDA:** Training works on CPU (slower). Install CUDA toolkit for GPU support.

## Model Version Control

The script automatically commits trained models to GitHub:

```bash
# After training, commit and push models
./setup_and_run.sh save
```

Models are saved to: `save/AASCE_interactive_keypoint_estimation/model.pth`

The script will:
1. Add the `save/` directory to git
2. Commit with timestamp and training info
3. Ask to push to GitHub (or auto-push in non-interactive mode with `-y`)

**Note:** Large model files (`.pth`, `.pt`, etc.) use Git LFS (configured in `.gitattributes`).

